import numpy as np
def build_tree(depth, sen):
    """该函数直接复制自原作者代码
    """
    assert len(depth) == len(sen)
    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = np.argmin(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree



#>>>>>>>>>>>>>>>>>example>>>>>>>>>>>>>>

# def parse_sent(s):
#     s = tokenize(s)
#     sid = np.array([string2id(s)[:-1]])
#     sl = lm_f([sid])[0][0][1:]
#     # 用json.dumps的indent功能，最简单地可视化效果
#     return json.dumps(build_tree(sl, s), indent=4, ensure_ascii=False)