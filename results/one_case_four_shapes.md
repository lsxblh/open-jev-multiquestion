# 同一个 case 在四份数据集里的形态

`group_id = customer-control-v1:42:185`

```text
Customer Harbor-bc9005dbda: I need access restored after changing the email address used to sign in. That is the request I want you to handle first.
Agent: Could you add the details?
Customer: Well, that was something. I will keep using the service and renew my subscription.
```


## typed_full: **1 条**

- 第 0 条：questions = `['category', 'bug_severity', 'has_reproducible_steps', 'refund_requested', 'frustration', 'churn_likelihood_level']`，gold label = `{'category': 'account', 'bug_severity': '0', 'has_reproducible_steps': 'false', 'refund_requested': 'false', 'frustration': '0', 'churn_likelihood_level': '0'}`，n_questions = 6

state 是否被原样保留: True


## typed_split: **6 条**

- 第 0 条：questions = `['category']`，gold label = `{'category': 'account'}`，n_questions = 1
- 第 1 条：questions = `['bug_severity']`，gold label = `{'bug_severity': '0'}`，n_questions = 1
- 第 2 条：questions = `['has_reproducible_steps']`，gold label = `{'has_reproducible_steps': 'false'}`，n_questions = 1
- 第 3 条：questions = `['refund_requested']`，gold label = `{'refund_requested': 'false'}`，n_questions = 1
- 第 4 条：questions = `['frustration']`，gold label = `{'frustration': '0'}`，n_questions = 1
- 第 5 条：questions = `['churn_likelihood_level']`，gold label = `{'churn_likelihood_level': '0'}`，n_questions = 1

state 是否被原样保留: True


## hybrid: **1 条**

- 第 0 条：questions = `['category', 'bug_severity', 'has_reproducible_steps', 'refund_requested', 'frustration', 'churn_likelihood_level']`，gold label = `{'category': 'account', 'bug_severity': '0', 'has_reproducible_steps': 'false', 'refund_requested': 'false', 'frustration': '0', 'churn_likelihood_level': '0'}`，n_questions = 6

state 是否被原样保留: True


## hybrid_v2: **1 条**

- 第 0 条：questions = `['category', 'bug_severity', 'has_reproducible_steps', 'refund_requested', 'frustration', 'churn_likelihood_level']`，gold label = `{'category': 'account', 'bug_severity': '0', 'has_reproducible_steps': 'false', 'refund_requested': 'false', 'frustration': '0', 'churn_likelihood_level': '0'}`，n_questions = 6

state 是否被原样保留: True
