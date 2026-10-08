"""No TensorFlow required for data-window safety tests."""
import numpy as np
import pytest
from neural_tuning import build_windows, neural_walk_forward
from quant_research import WalkForwardConfig


def test_neural_window_ends_at_decision_time():
    feature=np.arange(40*2,dtype=float).reshape(40,2)
    target=np.arange(40,dtype=float)
    X,y,ids=build_windows(feature,target,np.array([10,20]),lookback=5)
    assert X.shape==(2,5,2)
    assert np.all(X[0,-1]==feature[10])
    assert np.all(X[1,-1]==feature[20])
    assert y.tolist()==[10,20]
    assert ids.tolist()==[10,20]
    assert not np.isin(feature[11],X[0]).all()


def test_neural_window_does_not_include_future_change():
    feature=np.arange(40*2,dtype=float).reshape(40,2)
    y=np.zeros(40)
    x,_,_=build_windows(feature,y,np.array([10]),5)
    altered=feature.copy()
    altered[11:,:]=1000000
    later,_,_=build_windows(altered,y,np.array([10]),5)
    np.testing.assert_array_equal(x,later)


def test_empty_windows_and_invalid_horizon():
    x=np.ones((20,3))
    y=np.ones(20)
    empty,_,_=build_windows(x,y,np.array([2]),10)
    assert empty.shape[0]==0
    with pytest.raises(ValueError):
        build_windows(x,y,np.array([10]),1)


def test_invalid_neural_epochs_rejected_before_tensorflow_import():
    with pytest.raises(ValueError):
        neural_walk_forward(None,WalkForwardConfig(),max_epochs=0)
