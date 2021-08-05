import numpy as np
from icecream import ic
from ail.common.running_stats import RunningStats


def test_running_stat_fake_data():
    for shp in ((), (3,), (3, 4)):
        li = []
        rs = RunningStats(shp)
        for _ in range(5):
            val = np.random.randn(*shp)
            rs.push(val)
            li.append(val)
            m = np.mean(li, axis=0)
            assert np.allclose(rs.mean, m)
            v = np.square(m) if (len(li) == 1) else np.var(li, ddof=1, axis=0)
            assert np.allclose(rs.var, v)


def test_running_stats_gym():
    import gym

    env = gym.make("Pendulum-v0")
    obs = env.reset()
    ic(obs.shape)
    shp = obs.shape
    rs = RunningStats(shp)
    lst = []
    for i in range(100):
        act = env.action_space.sample()
        obs, *_ = env.step(act)

        lst.append(obs)
        rs.push(obs)
    lst = np.asarray(lst)
    ic(lst.shape)
    m = np.mean(lst, axis=0)
    v = np.square(m) if (len(lst) == 1) else np.var(lst, ddof=1, axis=0)
    ic(m, v)
    ic(rs.mean, rs.var)
    np.testing.assert_allclose(rs.mean, m)
    np.testing.assert_allclose(rs.var, v)
