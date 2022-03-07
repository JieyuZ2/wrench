ABSTAIN = -1
HAM = 0
SPAM = 1
@labeling_function()
def lf_text_len(x):
    return HAM if x.text_len > 177 else ABSTAIN
@labeling_function()
def lf_avg_count(x):
    return HAM if x["count"] > 2 else ABSTAIN
@labeling_function()
def lf_time_count(x):
    return  HAM  if x.date_std > 200 else ABSTAIN
@labeling_function()
def check(x):
    return SPAM if "check" in x.text.lower() else ABSTAIN

@labeling_function()
def check_out(x):
    return SPAM if "check out" in x.text.lower() else ABSTAIN
@labeling_function()
def regex_check_out(x):
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN
