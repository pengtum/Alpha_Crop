"""
不变性核查（toy ctx，N=3000，k∈{50,200}，seed∈{3,11}）：
 A 无关字段扰动：T_s2/F_s2/F_ae/F_xy/T_nuts3/F_country 换成噪声，P_T_loco/dom/P_F_oof 置 None，G 配置下 T_xy 也换噪声
 B 几何类方法对 P_T 替换是否不变
 C 预测类方法对 T_ae 替换是否不变
 D T_ae 整体缩放 ×7.3
 E T_ae 平移（每维加上 3 倍该维标准差的常数）
 F T_ae 逐列翻号（PCA 符号任意）
 G P_T 温度变化（s=0.6 与 1.6）
 H XY 参照：把居中/不居中的公里坐标放进 T_ae（加 5000 km 偏移）
输出：每项“逐位一致 / 选中集合重合比例”。
"""
import sys
import copy
import numpy as np

CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
import samplers.base as B  # noqa: E402
import samplers.designs, samplers.uncertainty, samplers.coverage, samplers.transfer  # noqa: E402,F401

GEOM = ['lpm2', 'fscs', 'kcenter', 'typiclust', 'kherding', 'gist_maxmin']
PRED = ['margin', 'predstrat_srs']
BOTH = ['gist_margin', 'predstrat', 'badge', 'fass_quota']
ALL = ['srs'] + GEOM + PRED + BOTH
HP = {'lpm2': {'geo_w': 0.0}, 'predstrat': {'geo_w': 0.0}, 'predstrat_srs': {'geo_w': 0.0}}


def sel(c, name, k, seed):
    c = copy.copy(c)
    c.meta = {}
    return B.REGISTRY[name](c, k, seed, **HP.get(name, {}))


def cmp(a, b):
    if np.array_equal(a, b):
        return 'same'
    return f'{len(np.intersect1d(a, b)) / len(a):.2f}'


def with_(c, **kw):
    c2 = copy.copy(c)
    for k_, v in kw.items():
        setattr(c2, k_, v)
    return c2


def main():
    base = B.make_toy_ctx(N=3000, seed=0)
    base.oracle = None
    r = np.random.default_rng(99)
    N, nF = base.N, len(base.F_y)
    noise = dict(T_s2=r.normal(size=(N, 32)).astype(np.float32), F_s2=r.normal(size=(nF, 32)).astype(np.float32),
                 F_ae=r.normal(size=(nF, 32)).astype(np.float32), F_xy=r.normal(size=(nF, 2)),
                 T_nuts3=r.integers(0, 7, N), F_country=r.integers(0, 3, nF), P_T_loco=None, dom=None, P_F_oof=None,
                 T_xy=r.uniform(-1e3, 1e3, size=(N, 2)))
    P2 = r.dirichlet(np.ones(10) * 0.3, size=N)
    T2 = r.normal(size=(N, 32)).astype(np.float32) * np.linspace(3, 0.3, 32).astype(np.float32)
    sd = base.T_ae.std(0)
    logP = np.log(np.clip(base.P_T, 1e-12, None))

    def temp(s):
        z = s * logP
        z -= z.max(1, keepdims=True)
        q = np.exp(z)
        return q / q.sum(1, keepdims=True)
    flips = np.where(r.random(32) < 0.5, -1, 1).astype(np.float32)
    xy = base.T_xy.copy()
    tests = {
        'A_无关字段噪声': with_(base, **noise),
        'B_换P_T': with_(base, P_T=P2),
        'C_换T_ae': with_(base, T_ae=T2),
        'D_缩放x7.3': with_(base, T_ae=(base.T_ae * 7.3).astype(np.float32)),
        'E_平移3sd': with_(base, T_ae=(base.T_ae + 3 * sd).astype(np.float32)),
        'F_翻号': with_(base, T_ae=(base.T_ae * flips).astype(np.float32)),
        'G_温度0.6': with_(base, P_T=temp(0.6)),
        'G_温度1.6': with_(base, P_T=temp(1.6)),
    }
    xy_c = with_(base, T_ae=(xy - xy.mean(0)).astype(np.float64))
    xy_off = with_(base, T_ae=(xy + 5000.0).astype(np.float64))
    rows = []
    for name in ALL:
        for k in (50, 200):
            for seed in (3, 11):
                ref = sel(base, name, k, seed)
                row = {'method': name, 'k': k, 'seed': seed}
                for t, c in tests.items():
                    try:
                        row[t] = cmp(ref, sel(c, name, k, seed))
                    except Exception as e:  # noqa: BLE001
                        row[t] = f'ERR {type(e).__name__}'
                if name in GEOM:
                    try:
                        a = sel(xy_c, name, k, seed)
                        b = sel(xy_off, name, k, seed)
                        row['H_XY居中vs偏移5000km'] = cmp(a, b)
                    except Exception as e:  # noqa: BLE001
                        row['H_XY居中vs偏移5000km'] = f'ERR {type(e).__name__}: {e}'
                rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(df.to_string(index=False))
    df.to_csv('invariance.csv', index=False)


if __name__ == '__main__':
    main()
