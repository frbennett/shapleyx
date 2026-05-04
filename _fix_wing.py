import json

# ----- Fix wing_weight.ipynb cell 17 -----
with open('Examples/wing_weight.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][17]['source'] = [
    '# Published first-order Sobol indices (OpenTURNS, Saltelli 2002)\n',
    '# From: https://openturns.github.io/openturns/latest\n',
    'ref_sobol_first = np.array([\n',
    '    0.130315,    # Sw\n',
    '    2.94e-06,    # Wfw\n',
    '    0.228153,    # A\n',
    '    0.0,         # Lambda\n',
    '    8.25e-05,    # q\n',
    '    0.001803,    # l (taper ratio)\n',
    '    0.135002,    # tc\n',
    '    0.412794,    # Nz\n',
    '    0.088332,    # Wdg\n',
    '    0.003516,    # Wp\n',
    '])\n',
    '\n',
    '# Our RS-HDMR Sobol first-order indices\n',
    "our_sobol_mc = mc_surrogate['sobol_first'].values\n",
    '\n',
    'validation = pd.DataFrame({\n',
    "    'Variable': labels,\n",
    "    'MC Sobol S_i': our_sobol_mc,\n",
    "    'RS-HDMR Sobol S_i': filtered_df['index'],\n",
    "    'Published S_i (OpenTURNS)': ref_sobol_first,\n",
    '})\n',
    'validation'
]

with open('Examples/wing_weight.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Also fix the copy in docs/tutorials
with open('docs/tutorials/wing_weight.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
nb2['cells'][17] = nb['cells'][17]
with open('docs/tutorials/wing_weight.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)

print('Fixed cell 17 bug: our_sobol → our_sobol_mc')
