from test.unit.test_helpers import TensorTestCase

import torch

from joeynmt.attention import BahdanauAttention, LuongAttention


class TestBahdanauAttention(TensorTestCase):

    def setUp(self):
        self.key_size = 3
        self.query_size = 5
        self.hidden_size = 7
        seed = 42
        torch.manual_seed(seed)
        self.bahdanau_att = BahdanauAttention(
            hidden_size=self.hidden_size,
            key_size=self.key_size,
            query_size=self.query_size,
        )

    def test_bahdanau_attention_size(self):
        self.assertIsNone(self.bahdanau_att.key_layer.bias)  # no bias
        self.assertIsNone(self.bahdanau_att.query_layer.bias)  # no bias
        self.assertEqual(
            self.bahdanau_att.key_layer.weight.shape,
            torch.Size([self.hidden_size, self.key_size]),
        )
        self.assertEqual(
            self.bahdanau_att.query_layer.weight.shape,
            torch.Size([self.hidden_size, self.query_size]),
        )
        self.assertEqual(
            self.bahdanau_att.energy_layer.weight.shape,
            torch.Size([1, self.hidden_size]),
        )
        self.assertIsNone(self.bahdanau_att.energy_layer.bias)

    def test_bahdanau_forward(self):
        src_length = 5
        trg_length = 4
        batch_size = 6
        queries = torch.rand(size=(batch_size, trg_length, self.query_size))
        keys = torch.rand(size=(batch_size, src_length, self.key_size))
        mask = torch.ones(size=(batch_size, 1, src_length)).byte()
        # introduce artificial padding areas
        mask[0, 0, -3:] = 0
        mask[1, 0, -2:] = 0
        mask[4, 0, -1:] = 0
        for t in range(trg_length):
            c, att = None, None
            try:
                # should raise an AssertionException (missing pre-computation)
                query = queries[:, t, :].unsqueeze(1)
                c, att = self.bahdanau_att(query=query, mask=mask, values=keys)
            except AssertionError:
                pass
            self.assertIsNone(c)
            self.assertIsNone(att)

        # now with pre-computation
        self.bahdanau_att.compute_proj_keys(keys=keys)
        self.assertIsNotNone(self.bahdanau_att.proj_keys)
        self.assertEqual(
            self.bahdanau_att.proj_keys.shape,
            torch.Size([batch_size, src_length, self.hidden_size]),
        )
        contexts = []
        attention_probs = []
        for t in range(trg_length):
            c, att = None, None
            try:
                # should not raise an AssertionException
                query = queries[:, t, :].unsqueeze(1)
                c, att = self.bahdanau_att(query=query, mask=mask, values=keys)
            except AssertionError:
                self.fail()
            self.assertIsNotNone(c)
            self.assertIsNotNone(att)
            self.assertEqual(
                self.bahdanau_att.proj_query.shape,
                torch.Size([batch_size, 1, self.hidden_size]),
            )
            contexts.append(c)
            attention_probs.append(att)
        self.assertEqual(len(attention_probs), trg_length)
        self.assertEqual(len(contexts), trg_length)
        contexts = torch.cat(contexts, dim=1)
        attention_probs = torch.cat(attention_probs, dim=1)
        self.assertEqual(contexts.shape,
                         torch.Size([batch_size, trg_length, self.key_size]))
        self.assertEqual(
            attention_probs.shape,
            torch.Size([batch_size, trg_length, src_length]),
        )
        contexts_target = torch.Tensor([
            [
                [0.5080, 0.5832, 0.5614],
                [0.5096, 0.5816, 0.5596],
                [0.5092, 0.5820, 0.5601],
                [0.5079, 0.5833, 0.5615],
            ],
            [
                [0.4709, 0.5817, 0.3091],
                [0.4720, 0.5793, 0.3063],
                [0.4704, 0.5825, 0.3102],
                [0.4709, 0.5814, 0.3090],
            ],
            [
                [0.4394, 0.4482, 0.6526],
                [0.4390, 0.4475, 0.6522],
                [0.4391, 0.4479, 0.6538],
                [0.4391, 0.4479, 0.6533],
            ],
            [
                [0.5283, 0.3441, 0.3938],
                [0.5297, 0.3457, 0.3956],
                [0.5306, 0.3466, 0.3966],
                [0.5274, 0.3431, 0.3926],
            ],
            [
                [0.4079, 0.4145, 0.2439],
                [0.4064, 0.4156, 0.2445],
                [0.4077, 0.4147, 0.2439],
                [0.4067, 0.4153, 0.2444],
            ],
            [
                [0.5649, 0.5749, 0.4960],
                [0.5660, 0.5763, 0.4988],
                [0.5658, 0.5754, 0.4984],
                [0.5662, 0.5766, 0.4991],
            ],
        ])
        self.assertTensorAlmostEqual(contexts_target, contexts)

        attention_probs_targets = torch.Tensor([
            [
                [0.4904, 0.5096, 0.0000, 0.0000, 0.0000],
                [0.4859, 0.5141, 0.0000, 0.0000, 0.0000],
                [0.4871, 0.5129, 0.0000, 0.0000, 0.0000],
                [0.4906, 0.5094, 0.0000, 0.0000, 0.0000],
            ],
            [
                [0.3314, 0.3278, 0.3408, 0.0000, 0.0000],
                [0.3337, 0.3230, 0.3433, 0.0000, 0.0000],
                [0.3301, 0.3297, 0.3402, 0.0000, 0.0000],
                [0.3312, 0.3275, 0.3413, 0.0000, 0.0000],
            ],
            [
                [0.1977, 0.2047, 0.2040, 0.1936, 0.1999],
                [0.1973, 0.2052, 0.2045, 0.1941, 0.1988],
                [0.1987, 0.2046, 0.2046, 0.1924, 0.1996],
                [0.1984, 0.2047, 0.2044, 0.1930, 0.1995],
            ],
            [
                [0.1963, 0.2041, 0.2006, 0.1942, 0.2047],
                [0.1954, 0.2065, 0.2011, 0.1934, 0.2036],
                [0.1947, 0.2074, 0.2014, 0.1928, 0.2038],
                [0.1968, 0.2028, 0.2006, 0.1949, 0.2049],
            ],
            [
                [0.2455, 0.2414, 0.2588, 0.2543, 0.0000],
                [0.2450, 0.2447, 0.2566, 0.2538, 0.0000],
                [0.2458, 0.2417, 0.2586, 0.2540, 0.0000],
                [0.2452, 0.2438, 0.2568, 0.2542, 0.0000],
            ],
            [
                [0.1999, 0.1888, 0.1951, 0.2009, 0.2153],
                [0.2035, 0.1885, 0.1956, 0.1972, 0.2152],
                [0.2025, 0.1885, 0.1950, 0.1980, 0.2159],
                [0.2044, 0.1884, 0.1955, 0.1970, 0.2148],
            ],
        ])
        self.assertTensorAlmostEqual(attention_probs_targets, attention_probs)

    def test_bahdanau_precompute_None(self):
        self.assertIsNone(self.bahdanau_att.proj_keys)
        self.assertIsNone(self.bahdanau_att.proj_query)

    def test_bahdanau_precompute(self):
        src_length = 5
        batch_size = 6
        keys = torch.rand(size=(batch_size, src_length, self.key_size))
        self.bahdanau_att.compute_proj_keys(keys=keys)
        proj_keys_targets = torch.tensor([
            [
                [0.4042, 0.1373, 0.3308, 0.2317, 0.3011, 0.2978, -0.0975],
                [0.4740, 0.4829, -0.0853, -0.2634, 0.4623, 0.0333, -0.2702],
                [0.4540, 0.0645, 0.6046, 0.4632, 0.3459, 0.4631, -0.0919],
                [0.4744, 0.5098, -0.2441, -0.3713, 0.4265, -0.0407, -0.2527],
                [0.0314, 0.1189, 0.3825, 0.1119, 0.2548, 0.1239, -0.1921],
            ],
            [
                [0.7057, 0.2725, 0.2426, 0.1979, 0.4285, 0.3727, -0.1126],
                [0.3967, 0.0223, 0.3664, 0.3488, 0.2107, 0.3531, -0.0095],
                [0.4311, 0.4695, 0.3035, -0.0640, 0.5914, 0.1713, -0.3695],
                [0.0797, 0.1038, 0.3847, 0.1476, 0.2486, 0.1568, -0.1672],
                [0.3379, 0.3671, 0.3622, 0.0166, 0.5097, 0.1845, -0.3207],
            ],
            [
                [0.4051, 0.4552, -0.0709, -0.2616, 0.4339, 0.0126, -0.2682],
                [0.5379, 0.5037, 0.0074, -0.2046, 0.5243, 0.0969, -0.2953],
                [0.0250, 0.0544, 0.3859, 0.1679, 0.1976, 0.1471, -0.1392],
                [0.1880, 0.2725, 0.1849, -0.0598, 0.3383, 0.0693, -0.2329],
                [0.0759, 0.1006, 0.0955, -0.0048, 0.1361, 0.0400, -0.0913],
            ],
            [
                [-0.0207, 0.1266, 0.5529, 0.1728, 0.3192, 0.1611, -0.2560],
                [0.5713, 0.2364, 0.0718, 0.0801, 0.3141, 0.2455, -0.0729],
                [0.1574, 0.1162, 0.3591, 0.1572, 0.2602, 0.1838, -0.1510],
                [0.1357, 0.0192, 0.1817, 0.1391, 0.1037, 0.1389, -0.0277],
                [0.3088, 0.2804, 0.2024, -0.0045, 0.3680, 0.1386, -0.2127],
            ],
            [
                [0.1181, 0.0899, 0.1139, 0.0329, 0.1390, 0.0744, -0.0758],
                [0.0713, 0.2682, 0.4111, 0.0129, 0.4044, 0.0985, -0.3177],
                [0.5340, 0.1713, 0.5365, 0.3679, 0.4262, 0.4373, -0.1456],
                [0.3902, -0.0242, 0.4498, 0.4313, 0.1997, 0.4012, 0.0075],
                [0.1764, 0.1531, -0.0564, -0.0876, 0.1390, 0.0129, -0.0714],
            ],
            [
                [0.3772, 0.3725, 0.3053, -0.0012, 0.4982, 0.1808, -0.3006],
                [0.4391, -0.0472, 0.3379, 0.4136, 0.1434, 0.3918, 0.0687],
                [0.3697, 0.2313, 0.4745, 0.2100, 0.4348, 0.3000, -0.2242],
                [0.8427, 0.3705, 0.1227, 0.1079, 0.4890, 0.3604, -0.1305],
                [0.3526, 0.3477, 0.1473, -0.0740, 0.4132, 0.1138, -0.2452],
            ],
        ])
        self.assertTensorAlmostEqual(proj_keys_targets, self.bahdanau_att.proj_keys)


class TestLuongAttention(TensorTestCase):

    def setUp(self):
        self.addTypeEqualityFunc(
            torch.Tensor,
            lambda x, y, msg: self.failureException(msg)
            if not torch.equal(x, y) else True,
        )
        self.key_size = 3
        self.query_size = 5
        self.hidden_size = self.query_size
        seed = 42
        torch.manual_seed(seed)
        self.luong_att = LuongAttention(hidden_size=self.hidden_size,
                                        key_size=self.key_size)

    def test_luong_attention_size(self):
        self.assertIsNone(self.luong_att.key_layer.bias)  # no bias
        self.assertEqual(
            self.luong_att.key_layer.weight.shape,
            torch.Size([self.hidden_size, self.key_size]),
        )

    def test_luong_attention_forward(self):
        src_length = 5
        trg_length = 4
        batch_size = 6
        queries = torch.rand(size=(batch_size, trg_length, self.query_size))
        keys = torch.rand(size=(batch_size, src_length, self.key_size))
        mask = torch.ones(size=(batch_size, 1, src_length)).byte()
        # introduce artificial padding areas
        mask[0, 0, -3:] = 0
        mask[1, 0, -2:] = 0
        mask[4, 0, -1:] = 0
        for t in range(trg_length):
            c, att = None, None
            try:
                # should raise an AssertionException (missing pre-computation)
                query = queries[:, t, :].unsqueeze(1)
                c, att = self.luong_att(query=query, mask=mask, values=keys)
            except AssertionError:
                pass
            self.assertIsNone(c)
            self.assertIsNone(att)

        # now with pre-computation
        self.luong_att.compute_proj_keys(keys=keys)
        self.assertIsNotNone(self.luong_att.proj_keys)
        self.assertEqual(
            self.luong_att.proj_keys.shape,
            torch.Size([batch_size, src_length, self.hidden_size]),
        )
        contexts = []
        attention_probs = []
        for t in range(trg_length):
            c, att = None, None
            try:
                # should not raise an AssertionException
                query = queries[:, t, :].unsqueeze(1)
                c, att = self.luong_att(query=query, mask=mask, values=keys)
            except AssertionError:
                self.fail()
            self.assertIsNotNone(c)
            self.assertIsNotNone(att)
            contexts.append(c)
            attention_probs.append(att)
        self.assertEqual(len(attention_probs), trg_length)
        self.assertEqual(len(contexts), trg_length)
        contexts = torch.cat(contexts, dim=1)
        attention_probs = torch.cat(attention_probs, dim=1)
        self.assertEqual(contexts.shape,
                         torch.Size([batch_size, trg_length, self.key_size]))
        self.assertEqual(
            attention_probs.shape,
            torch.Size([batch_size, trg_length, src_length]),
        )
        context_targets = torch.Tensor([
            [
                [0.5347, 0.2918, 0.4707],
                [0.5062, 0.2657, 0.4117],
                [0.4969, 0.2572, 0.3926],
                [0.5320, 0.2893, 0.4651],
            ],
            [
                [0.5210, 0.6707, 0.4343],
                [0.5111, 0.6809, 0.4274],
                [0.5156, 0.6622, 0.4274],
                [0.5046, 0.6634, 0.4175],
            ],
            [
                [0.4998, 0.5570, 0.3388],
                [0.4949, 0.5357, 0.3609],
                [0.4982, 0.5208, 0.3468],
                [0.5013, 0.5474, 0.3503],
            ],
            [
                [0.5911, 0.6944, 0.5319],
                [0.5964, 0.6899, 0.5257],
                [0.6161, 0.6771, 0.5042],
                [0.5937, 0.7011, 0.5330],
            ],
            [
                [0.4439, 0.5916, 0.3691],
                [0.4409, 0.5970, 0.3762],
                [0.4446, 0.5845, 0.3659],
                [0.4417, 0.6157, 0.3796],
            ],
            [
                [0.4581, 0.4343, 0.5151],
                [0.4493, 0.4297, 0.5348],
                [0.4399, 0.4265, 0.5419],
                [0.4833, 0.4570, 0.4855],
            ],
        ])
        self.assertTensorAlmostEqual(context_targets, contexts)
        attention_probs_targets = torch.Tensor([
            [
                [0.3238, 0.6762, 0.0000, 0.0000, 0.0000],
                [0.4090, 0.5910, 0.0000, 0.0000, 0.0000],
                [0.4367, 0.5633, 0.0000, 0.0000, 0.0000],
                [0.3319, 0.6681, 0.0000, 0.0000, 0.0000],
            ],
            [
                [0.2483, 0.3291, 0.4226, 0.0000, 0.0000],
                [0.2353, 0.3474, 0.4174, 0.0000, 0.0000],
                [0.2725, 0.3322, 0.3953, 0.0000, 0.0000],
                [0.2803, 0.3476, 0.3721, 0.0000, 0.0000],
            ],
            [
                [0.1955, 0.1516, 0.2518, 0.1466, 0.2546],
                [0.2220, 0.1613, 0.2402, 0.1462, 0.2303],
                [0.2074, 0.1953, 0.2142, 0.1536, 0.2296],
                [0.2100, 0.1615, 0.2434, 0.1376, 0.2475],
            ],
            [
                [0.2227, 0.2483, 0.1512, 0.1486, 0.2291],
                [0.2210, 0.2331, 0.1599, 0.1542, 0.2318],
                [0.2123, 0.1808, 0.1885, 0.1702, 0.2482],
                [0.2233, 0.2479, 0.1435, 0.1433, 0.2421],
            ],
            [
                [0.2475, 0.2482, 0.2865, 0.2178, 0.0000],
                [0.2494, 0.2410, 0.2976, 0.2120, 0.0000],
                [0.2498, 0.2449, 0.2778, 0.2275, 0.0000],
                [0.2359, 0.2603, 0.3174, 0.1864, 0.0000],
            ],
            [
                [0.2362, 0.1929, 0.2128, 0.1859, 0.1723],
                [0.2230, 0.2118, 0.2116, 0.1890, 0.1646],
                [0.2118, 0.2251, 0.2039, 0.1891, 0.1700],
                [0.2859, 0.1874, 0.2083, 0.1583, 0.1601],
            ],
        ])
        self.assertTensorAlmostEqual(attention_probs_targets, attention_probs)

    def test_luong_precompute_None(self):
        self.assertIsNone(self.luong_att.proj_keys)

    def test_luong_precompute(self):
        src_length = 5
        batch_size = 6
        keys = torch.rand(size=(batch_size, src_length, self.key_size))
        self.luong_att.compute_proj_keys(keys=keys)
        proj_keys_targets = torch.Tensor([
            [
                [0.5362, 0.1826, 0.4716, 0.3245, 0.4122],
                [0.3819, 0.0934, 0.2750, 0.2311, 0.2378],
                [0.2246, 0.2934, 0.3999, 0.0519, 0.4430],
                [0.1271, 0.0636, 0.2444, 0.1294, 0.1659],
                [0.3494, 0.0372, 0.1326, 0.1908, 0.1295],
            ],
            [
                [0.3363, 0.5984, 0.2090, -0.2695, 0.6584],
                [0.3098, 0.3608, 0.3623, 0.0098, 0.5004],
                [0.6133, 0.2568, 0.4264, 0.2688, 0.4716],
                [0.4058, 0.1438, 0.3043, 0.2127, 0.2971],
                [0.6604, 0.3490, 0.5228, 0.2593, 0.5967],
            ],
            [
                [0.4224, 0.1182, 0.4883, 0.3403, 0.3458],
                [0.4257, 0.3757, -0.1431, -0.2208, 0.3383],
                [0.0681, 0.2540, 0.4165, 0.0269, 0.3934],
                [0.5341, 0.3288, 0.3937, 0.1532, 0.5132],
                [0.6244, 0.1647, 0.2378, 0.2548, 0.3196],
            ],
            [
                [0.2222, 0.3380, 0.2374, -0.0748, 0.4212],
                [0.4042, 0.1373, 0.3308, 0.2317, 0.3011],
                [0.4740, 0.4829, -0.0853, -0.2634, 0.4623],
                [0.4540, 0.0645, 0.6046, 0.4632, 0.3459],
                [0.4744, 0.5098, -0.2441, -0.3713, 0.4265],
            ],
            [
                [0.0314, 0.1189, 0.3825, 0.1119, 0.2548],
                [0.7057, 0.2725, 0.2426, 0.1979, 0.4285],
                [0.3967, 0.0223, 0.3664, 0.3488, 0.2107],
                [0.4311, 0.4695, 0.3035, -0.0640, 0.5914],
                [0.0797, 0.1038, 0.3847, 0.1476, 0.2486],
            ],
            [
                [0.3379, 0.3671, 0.3622, 0.0166, 0.5097],
                [0.4051, 0.4552, -0.0709, -0.2616, 0.4339],
                [0.5379, 0.5037, 0.0074, -0.2046, 0.5243],
                [0.0250, 0.0544, 0.3859, 0.1679, 0.1976],
                [0.1880, 0.2725, 0.1849, -0.0598, 0.3383],
            ],
        ])
        self.assertTensorAlmostEqual(proj_keys_targets, self.luong_att.proj_keys)
