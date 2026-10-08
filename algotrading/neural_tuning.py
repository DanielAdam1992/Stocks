"""Optional LSTM/GRU/1D-CNN walk-forward retraining on return labels.

This is a from-scratch architecture comparison, not a loaded/trusted legacy
Colab artifact. TensorFlow is imported only when running an experiment.
Signals as of close t are evaluated from open(t+1) to open(t+2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from quant_research import FEATURES, WalkForwardConfig, make_features, performance
from sector_runner import ALL_ASSETS, load_prices

# Same architecture families as legacy notebooks, with finite CPU-friendly sizes.
CANDIDATES=(
    ("LSTM", 22, 24, .001),
    ("GRU", 22, 24, .001),
    ("CNN", 22, 24, .001),
    ("LSTM", 66, 16, .0005),
    ("GRU", 66, 16, .0005),
    ("CNN", 66, 16, .0005),
)


def build_windows(scaled_features: np.ndarray, targets: np.ndarray,
                  indices: np.ndarray, lookback: int):
    """End-inclusive history: window for row i never includes row i+1."""
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    X=[]
    y=[]
    valid_indices=[]
    for i in indices:
        i=int(i)
        if i<lookback-1 or i>=len(scaled_features):
            continue
        X.append(scaled_features[i-lookback+1:i+1])
        y.append(targets[i])
        valid_indices.append(i)
    if not X:
        return (np.empty((0,lookback,scaled_features.shape[1])),
                np.empty((0,)), np.array([],dtype=int))
    return np.stack(X),np.asarray(y),np.asarray(valid_indices)


def neural_model(architecture: str, lookback: int, n_features: int,
                 units: int, learning_rate: float):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for neural training") from exc
    tf.keras.utils.set_random_seed(42)
    input_layer=tf.keras.layers.Input(shape=(lookback,n_features))
    if architecture=="LSTM":
        feature_layer=tf.keras.layers.LSTM(units)(input_layer)
    elif architecture=="GRU":
        feature_layer=tf.keras.layers.GRU(units)(input_layer)
    elif architecture=="CNN":
        conv=tf.keras.layers.Conv1D(units,kernel_size=3,padding="causal",
                                    activation="relu")(input_layer)
        feature_layer=tf.keras.layers.GlobalAveragePooling1D()(conv)
    else:
        raise ValueError("Unsupported architecture")
    hidden=tf.keras.layers.Dropout(.20)(feature_layer)
    final=tf.keras.layers.Dense(1)(hidden)
    model=tf.keras.Model(input_layer,final)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.Huber(delta=1.0))
    return model


def neural_walk_forward(prices: pd.DataFrame,
                        config: WalkForwardConfig=WalkForwardConfig(),
                        candidates=CANDIDATES,max_epochs=25,batch_size=32):
    if max_epochs<1 or batch_size<1:
        raise ValueError("Invalid training settings")
    config.validate()
    df=make_features(prices)
    n=len(df)
    max_lookback=max(c[1] for c in candidates)
    earliest=config.min_train+config.validation_size+2*config.purge_rows+max_lookback
    start=max(earliest,n-config.test_size)
    if start>=n:
        raise ValueError("Insufficient history for neural model")
    import tensorflow as tf
    predictions=[]
    selection_history=[]
    y=(100*df["target_return"]).to_numpy(float)
    for pos in range(start,n,config.retrain_every):
        # At signal date pos, label at pos-1 is not yet resolved.
        hist_end=pos-config.purge_rows
        val_start=hist_end-config.validation_size
        fit_end=val_start-config.purge_rows
        scaler=StandardScaler().fit(df.loc[:fit_end-1,list(FEATURES)])
        scaled=scaler.transform(df[list(FEATURES)])
        inner=[]
        for arch,lookback,units,lr in candidates:
            X_train,y_train,_=build_windows(
                scaled,y,np.arange(lookback-1,fit_end),lookback)
            X_valid,y_valid,_=build_windows(
                scaled,y,np.arange(val_start,hist_end),lookback)
            if len(X_train)<50 or len(X_valid)<20:
                continue
            model=neural_model(arch,lookback,len(FEATURES),units,lr)
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",patience=4,restore_best_weights=True)]
            history=model.fit(
                X_train,y_train,validation_data=(X_valid,y_valid),
                epochs=max_epochs,batch_size=batch_size,shuffle=False,
                callbacks=callbacks,verbose=0)
            estimate=model.predict(X_valid,verbose=0).reshape(-1)
            rmse=float(np.sqrt(np.mean((y_valid-estimate)**2)))/100.0
            inner.append({"architecture":arch,"lookback":lookback,
                          "units":units,"learning_rate":lr,"val_rmse":rmse,
                          "epochs":len(history.history["loss"]),
                          "_model":model})
        if not inner:
            raise ValueError("No neural candidate had sufficient training data")
        selected=min(inner,key=lambda x:x["val_rmse"])
        model=selected["_model"]
        positions=np.arange(pos,min(pos+config.retrain_every,n))
        X_test,_,ids=build_windows(scaled,y,positions,selected["lookback"])
        out=df.iloc[ids].copy()
        out["predicted_return"]=model.predict(X_test,verbose=0).reshape(-1)/100.0
        out["model"]=selected["architecture"]
        out["train_end_date"]=df.iloc[hist_end-1]["Date"]
        predictions.append(out)
        selection_history.append({
            "first_test_date":str(out["Date"].iloc[0].date()),
            "trained_before":str(df.iloc[hist_end-1]["Date"].date()),
            "selected":{k:v for k,v in selected.items() if k!="_model"},
            "validation_results":[{k:v for k,v in c.items() if k!="_model"} for c in inner]})
        tf.keras.backend.clear_session()
    return pd.concat(predictions,ignore_index=True),selection_history


def main():
    ap=argparse.ArgumentParser(description="Optional compute-heavy neural architecture study")
    ap.add_argument("ticker",choices=sorted(ALL_ASSETS))
    ap.add_argument("--data-dir",default=None)
    ap.add_argument("--oos",type=int,default=252)
    ap.add_argument("--retrain",type=int,default=63)
    ap.add_argument("--epochs",type=int,default=25)
    ap.add_argument("--out",default="output/neural_study")
    args=ap.parse_args()
    raw=load_prices(args.ticker,source_dir=args.data_dir)
    c=WalkForwardConfig(test_size=args.oos,retrain_every=args.retrain)
    predictions,choices=neural_walk_forward(raw,c,max_epochs=args.epochs)
    details,stats=performance(predictions,c)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    details.to_csv(out/f"{args.ticker}_neural_oos.csv",index=False)
    result={"architecture_comparison_from_scratch":True,"ticker":args.ticker,
            "research_only":True,"stats":stats,"selections":choices,
            "warning":"Legacy checkpoint weights NOT loaded; neural performance unverified until run"}
    (out/f"{args.ticker}_neural.json").write_text(
        json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
