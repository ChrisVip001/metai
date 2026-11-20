# 真实日常句逐句双向翻译

✓ 为整句通过，△ 为指定难点正确但整句有误，✗ 为两项均未通过。小语料、广泛采样、难点采样各训练 1,536 步。[原答、各组逐句理由及 token ID](daily-v4.json)

## 不能／做不到

| 原句 | 参考译文 | 上一版 | 384 句对 | 广泛采样 | 难点采样 | 难点组判定依据 |
| --- | --- | --- | --- | --- | --- | --- |
| We can't talk here. | 我们不能在这里谈。 | ✗ 我们提供。 | ✗ 我們有說話。 | ✗ 我們不論這裡。 | ✗ 我们不能跟你说！ | 保留不能说，把地点限制改成了不能跟你说。 |
| 我们不能在这里谈。 | We can't talk here. | ✗ We don't want anything. | ✗ We want to see it. | ✗ We can't stay here till he comes. | ✗ We can't talk about it. | 把不能在这里谈改成不能谈此事，限制的范围不同。 |
| I can't remember. | 我想不起來。 | ✗ 我走不了。 | ✗ 我可以獨自做。 | ✗ 我無法去。 | ✓ 我不記得記得記得。 | “我不记得”传达无法回想，没有新增事实，但“记得”异常重复，流畅度差。 |
| 我想不起來。 | I can't remember. | ✗ I want to be a doctor. | ✗ I want to stay. | ✗ I would like to come. | ✗ I can't remember to come. | remember to come 指记得要来，给回忆改加了一个未来任务。 |
| I can't accept this gift. | 我不能收这份礼。 | ✗ 我不能預言會發生甚麼。 | ✗ 我可以獨自做這個。 | ✗ 我不接受這個女孩。 | ✗ 我不能破例。 | 不能收礼变成不能破例。 |
| 我不能收这份礼。 | I can't accept this gift. | ✗ I can't afford to play tennis. | ✗ I can't at home. | ✗ I can't stand this noise any longer. | ✗ I can't conceal with this. | conceal 是隐藏，未译出接受礼物。 |
| Tom can't ride a horse. | 湯姆不能騎馬。 | ✗ 汤姆不可能是真的。 | ✗ 汤姆不能打一點兒。 | ✗ 汤姆不会系领带。 | ✗ 湯姆不能自已。 | “不能自已”说的是控制不住自己。 |
| 湯姆不能騎馬。 | Tom can't ride a horse. | ✗ Tom can't be older. | ✗ Tom can't let anything. | ✗ Tom can't stay here. | ✗ Tom can't conceal number. | 骑马换成隐藏数字。 |
| Sorry, but I can't hear you very well. | 不好意思，我聽不清你說的話。 | ✗ 抱歉，我没钱。 | ✗ 抱歉我能知道得很抱歉。 | ✗ 對不起，但我無法讓你真的是正確。 | ✗ 对不起，但我情不自禁。 | 把听不清换成情不自禁。 |
| 不好意思，我聽不清你說的話。 | Sorry, but I can't hear you very well. | ✗ I can't decide if I'm happy or sad. | ✗ I wish I got the six hours. | ✗ I'm sorry to you don't know. | ✗ I'm sorry I'm sorry. | 只重复道歉，没有翻译听不清的原因。 |
| We can't escape. | 我們逃不出去。 | ✗ 我们提供水。 | ✗ 我們不能強迫害。 | ✗ 我們不能破例。 | ✗ 我们不能到欧洲。 | 逃不出去变成不能到欧洲。 |
| 我們逃不出去。 | We can't escape. | ✗ We don't have too much time. | ✗ We're having a good us. | ✗ We ought not to play. | ✗ We don't go out to play. | 变成不出去玩，遗漏受困而不能逃脱的意思。 |

<details><summary>本组原句署名</summary>

- CC-BY 2.0 (France) Attribution: tatoeba.org #1954967 (CK) & #5900366 (verdastelo9604)
- CC-BY 2.0 (France) Attribution: tatoeba.org #550246 (CK) & #910229 (Martha)
- CC-BY 2.0 (France) Attribution: tatoeba.org #1249224 (CM) & #1584522 (Gustav249)
- CC-BY 2.0 (France) Attribution: tatoeba.org #4666781 (CK) & #6105231 (verdastelo9604)
- CC-BY 2.0 (France) Attribution: tatoeba.org #2698827 (WestofEden) & #6333816 (verdastelo9604)
- CC-BY 2.0 (France) Attribution: tatoeba.org #2241048 (CK) & #6058637 (verdastelo9604)

</details>

## 一般否定

| 原句 | 参考译文 | 上一版 | 384 句对 | 广泛采样 | 难点采样 | 难点组判定依据 |
| --- | --- | --- | --- | --- | --- | --- |
| Tom didn't get much sleep. | 汤姆没怎么睡。 | ✗ 汤姆不点啤酒。 | ✗ 湯姆沒想冒犯一个。 | ✗ 湯姆沒有出現在這個大。 | ✗ 汤姆没有很多关于。 | “没有很多关于”没有给出所指的睡眠。 |
| 汤姆没怎么睡。 | Tom didn't get much sleep. | ✗ Tom can't sleep without a. | ✗ Tom have no one to come. | ✗ Tom isn't sleep. | ✗ Tom didn't sleep for an hour. | 加入原文没有的一小时，且没有清楚表达整体睡得少。 |
| It's not my fault. | 不是我的错。 | ✗ 不是我的。 | ✗ 它我的狗。 | ✗ 不是我的。 | ✗ 不是我的妻子。 | 把责任换成妻子。 |
| 不是我的错。 | It's not my fault. | ✗ It's not mine. | ✗ It's not mine. | ✗ It's not mine. | ✗ It's not my bag. | 换成否定包的所有权。 |
| This is not important. | 這個不重要。 | ✗ 這是不的。 | ✗ 這是很重要。 | ✗ 這個重要不重要。 | ✗ 這是個重要的。 | 把不重要译成重要。 |
| 這個不重要。 | This is not important. | ✗ This doesn't make sense. | ✗ This is not what he's good. | ✗ This isn't difficult. | ✗ This is not wanting. | 换成 wanting，未否定重要性。 |
| I'm not greedy. | 我不貪婪。 | ✗ 我不嫉妒。 | ✗ 我很不高兴。 | ✗ 我沒有出現在。 | ✗ 我不是死板的。 | 不死板不等于不贪婪。 |
| 我不貪婪。 | I'm not greedy. | ✗ I do not order juice. | ✗ I don't have to. | ✗ I can't buy annual. | ✗ I didn't feel like an hour. | 提到感觉和一小时，未译出不贪婪。 |
| Tom didn't write Mary back. | 湯姆沒回覆瑪麗。 | ✗ 湯姆不來不來。 | ✗ 湯姆沒寫瑪麗。 | ✗ 湯姆沒有把瑪麗不難。 | ✗ 湯姆沒有寫瑪麗。 | 辨出了两个人和写，但没有保留“回复”的关系。 |
| 湯姆沒回覆瑪麗。 | Tom didn't write Mary back. | ✗ Tom helped Mary buy everything. | ✗ Tom didn't mean to Mary. | ✗ Tom is later. | ✗ Tom is shy and Mary were both born. | 加入害羞和出生，未译出回复。 |
| I didn't mean to say that. | 我沒想那麼說。 | ✗ 我不想聽不到那麼。 | ✗ 我沒想弄壞的。 | ✓ 我沒想那麼說的。 | ✓ 我沒想那麼說。 | 与参考译文一致。 |
| 我沒想那麼說。 | I didn't mean to say that. | ✗ I'm not guilty of anything. | ✗ I didn't mean to that. | ✓ I didn't mean to say that. | ✓ I didn't mean to say that. | 与参考译文一致。 |

<details><summary>本组原句署名</summary>

- CC-BY 2.0 (France) Attribution: tatoeba.org #6439709 (CK) & #8591219 (easononizuka)
- CC-BY 2.0 (France) Attribution: tatoeba.org #321324 (CK) & #517562 (fucongcong)
- CC-BY 2.0 (France) Attribution: tatoeba.org #1385 (CK) & #59 (fucongcong)
- CC-BY 2.0 (France) Attribution: tatoeba.org #3267389 (Ricardo14) & #6333811 (verdastelo9604)
- CC-BY 2.0 (France) Attribution: tatoeba.org #3010330 (Hybrid) & #5965720 (verdastelo9604)
- CC-BY 2.0 (France) Attribution: tatoeba.org #6531676 (CK) & #13137955 (LeviHighway)

</details>

## 数量

| 原句 | 参考译文 | 上一版 | 384 句对 | 广泛采样 | 难点采样 | 难点组判定依据 |
| --- | --- | --- | --- | --- | --- | --- |
| My watch is five minutes slow. | 我的錶慢了五分鐘。 | ✗ 我的父親是真的。 | ✗ 我对他的态度变了。 | ✗ 我的手錶出了毛病。 | ✗ 我的手錶慢了這個機。 | 认出手表慢，未译出五分钟。 |
| 我的錶慢了五分鐘。 | My watch is five minutes slow. | ✗ My baby began crying, asking for milk. | ✗ My camera can shoot high-definition video. | ✗ My watch is collered with my watch. | ✗ My watch is waterproof. | 防水性不能表达手表慢五分钟。 |
| Believe it or not, she has three children. | 她有三个孩子，信不信由你。 | ✗ 請給她，但我把它丢了。 | ✗ 她的離開心。 | ✗ 請把她或不來，她是個孩子。 | ✗ 不再过，她不孩子们。 | 含孩子一词，三个以及她有孩子的关系都未表达。 |
| 她有三个孩子，信不信由你。 | Believe it or not, she has three children. | ✗ She's not, not, not injured. | ✗ She has three brothers of her work. | ✗ She has just able to you. | ✗ She's a child, she had no objection. | 译成她自己是孩子，还加上没有异议。 |
| I'm three years younger than you are. | 我比你小三歲。 | ✗ 我比你比你比你。 | ✗ 我很高兴你比你比你最好不要立刻做。 | ✗ 我年約年約年約年約年。 | △ 我比你多三歲。 | 保留三岁的年龄差，却把我比你年轻译成我比你年长。 |
| 我比你小三歲。 | I'm three years younger than you are. | ✗ I can't believe you did this. | ✗ I'm really surprised you got a prize. | ✗ I am three weeks at all right. | ✗ I three three three in three. | 只重复 three，没有年龄单位或比较。 |
| I eat meat three times a week. | 我每周吃肉三次。 | ✗ 我喝牛奶。 | ✗ 我吃午饭。 | ✗ 我吃午饭足够三个人吃。 | ✗ 我吃午饭足球员。 | 提到午饭和足球员，未表达频率。 |
| 我每周吃肉三次。 | I eat meat three times a week. | ✗ I like tea to eat snacks. | ✗ I have to eat lunchool in Boston. | ✗ I spend to eat lunch at all. | ✗ I have three weeks. | 三个星期是时长，不是每周吃三次。 |
| We ran three kilometers. | 我們跑了三公里。 | ✗ 我們通常飯後喝。 | ✗ 我們有充足的時間。 | ✗ 我們有兩張過。 | ✗ 我们三個小時候。 | 有数字三，未表达跑步或公里。 |
| 我們跑了三公里。 | We ran three kilometers. | ✗ We don't have anything. | ✗ We don't have to be in Boston. | ✗ We have three sandwork. | ✗ We have three parket on Sunday. | 有 three，但没有跑步或公里，还加上周日。 |
| I've climbed Mt. Fuji twice. | 我爬過兩次富士山。 | ✗ 我对外籍员工作。 | ✗ 我們有很容易治。 | ✗ 我們一點言會成總賴。 | ✗ 我很過去過人。 | 残句没有具体登山经历或两次。 |
| 我爬過兩次富士山。 | I've climbed Mt. Fuji twice. | ✗ I'm not to talk to the other day. | ✗ I went to the school in Boston. | ✗ I spent all the first time to the bus. | ✗ I was in Boston last year. | 变成去年在波士顿。 |

<details><summary>本组原句署名</summary>

- CC-BY 2.0 (France) Attribution: tatoeba.org #714329 (CM) & #714328 (egg0073)
- CC-BY 2.0 (France) Attribution: tatoeba.org #32854 (CK) & #429140 (fucongcong)
- CC-BY 2.0 (France) Attribution: tatoeba.org #6222208 (CK) & #864345 (Martha)
- CC-BY 2.0 (France) Attribution: tatoeba.org #1492179 (CK) & #6059525 (verdastelo9604)
- CC-BY 2.0 (France) Attribution: tatoeba.org #8075346 (CK) & #8387376 (wangchou)
- CC-BY 2.0 (France) Attribution: tatoeba.org #261486 (CK) & #793325 (Martha)

</details>

## 时间与体

| 原句 | 参考译文 | 上一版 | 384 句对 | 广泛采样 | 难点采样 | 难点组判定依据 |
| --- | --- | --- | --- | --- | --- | --- |
| We ate curry rice last night. | 我們昨晚吃了咖哩飯。 | ✗ 我們的友情會長久留存。 | ✗ 我們有很久時開。 | ✗ 我們去年約。 | ✗ 昨晚下大雨。 | 昨晚保留了，吃咖喱饭却变成下大雨，时间没有绑定原事件。 |
| 我們昨晚吃了咖哩飯。 | We ate curry rice last night. | ✗ We can't for coffee. | ✗ We're ready to coffee. | ✗ We ate ate pizz after colle. | △ We had lunch at home last night. | 正确把用餐放在昨晚，咖喱饭却变成午餐，并新增在家。 |
| I'm studying in the library. | 我正在图书馆学习。 | ✗ 我正在用現在這個問題。 | ✗ 我在沙發上砍下了一首車。 | ✗ 我在陽在學習裡。 | △ 在在在学习外見外。 | 可辨出“在学习”，但前后重复并生成无关字，图书馆和主语没有译清楚。 |
| 我正在图书馆学习。 | I'm studying in the library. | ✗ I'm drinking apple juice. | ✗ I'm glad to was there. | ✗ I'm the desk in a book at the desk. | △ I'm learning a book to learn. | 进行式 learning 保留正在学习，后半残缺且遗漏图书馆。 |
| Tom finished college last year. | 湯姆去年上完了大學。 | ✗ 湯姆卖咖啡。 | ✗ 汤姆去年去了波士顿了。 | ✗ 湯姆讓我們去年去了伦敦了。 | ✗ 湯姆去年去了咎世。 | 有去年，却未表达读完大学。 |
| 湯姆去年上完了大學。 | Tom finished college last year. | ✗ Tom will probably never be late again. | ✗ Tom visited to study to the stepper spray in the step. | ✗ Tom visited last year. | ✗ Tom visited to go to school last year. | 虽提到去年和学校，去上学并不表示完成大学学业。 |
| He left an hour ago. | 他一小时前走了。 | ✗ 他喝了三瓶啤酒。 | ✗ 他以前住在这儿。 | ✗ 他给自己一个人走。 | ✗ 他三年前死了。 | 变成三年前去世，时间和事件都错。 |
| 他一小时前走了。 | He left an hour ago. | ✗ It's not to eat something delicious. | ✗ I'm sorry to take my medic. | ✗ He has been to leave. | ✗ He is going to be out of fin. | 变成将来的不完整事件。 |
| They got here yesterday. | 他們昨天到這裡的。 | ✗ 他們在這裡這裡等。 | ✗ 他們親到這裡。 | ✗ 昨天这儿玩了。 | ✓ 他們昨天到這裡。 | 人物、昨天和到这里的事件齐全，句末不用“的”也自然。 |
| 他們昨天到這裡的。 | They got here yesterday. | ✗ They don't know anything about it. | ✗ They were persecuted. | ✗ Their yesterday's yesterday. | ✗ They were stayed in the yesterday. | 提到昨天，停留与到达不同。 |
| We're going to discuss the problem tomorrow. | 我們明天討論這個問題。 | ✗ 我們不要討去學校。 | ✗ 我們明天會在明天来。 | ✗ 我们明天要問了。 | ✗ 我们明天要去野餐。 | 把明天讨论问题改成明天去野餐。 |
| 我們明天討論這個問題。 | We're going to discuss the problem tomorrow. | ✗ We've never had this problem before. | ✗ We've got plenty of time. | ✗ We have to be painted this problem. | ✗ We're going to do tomorrow. | 只表达明天要做事，未说明讨论什么。 |

<details><summary>本组原句署名</summary>

- CC-BY 2.0 (France) Attribution: tatoeba.org #788259 (hrin) & #780272 (Martha)
- CC-BY 2.0 (France) Attribution: tatoeba.org #1488371 (marco87) & #6093308 (verdastelo9604)
- CC-BY 2.0 (France) Attribution: tatoeba.org #4664972 (CK) & #6075706 (verdastelo9604)
- CC-BY 2.0 (France) Attribution: tatoeba.org #73151 (CK) & #366884 (fucongcong)
- CC-BY 2.0 (France) Attribution: tatoeba.org #2737521 (CM) & #2737430 (egg0073)
- CC-BY 2.0 (France) Attribution: tatoeba.org #249413 (CK) & #6482313 (verdastelo9604)

</details>

## 其他日常表达

| 原句 | 参考译文 | 上一版 | 384 句对 | 广泛采样 | 难点采样 | 难点组判定依据 |
| --- | --- | --- | --- | --- | --- | --- |
| This bicycle is a gift for you. | 送你這輛自行車。 | ✗ 這個人過去。 | ✗ 這個你正在開的車的有机农场。 | ✗ 這輛自行車是我的。 | ✗ 這輛自行車是我的。 | 认出自行车，把赠给对方改为断言归我。 |
| 送你這輛自行車。 | This bicycle is a gift for you. | ✗ There's no milk in the. | ✗ This car he's driving. | ✗ It's bicycle. | ✗ The bicycle is bicy. | 认出自行车但未表达送给对方。 |
| Do you remember us? | 你还记得我们吗？ | ✗ 你經常早餐喝茶嗎？ | ✗ 你滑冰嗎？ | ✗ 你能帮我吗？ | ✗ 你記帳嗎？ | “记账”与“记得我们”含义不同。 |
| 你还记得我们吗？ | Do you remember us? | ✗ Why did you have tea? | ✗ What time do you go home? | ✗ You're invive don't you? | ✓ Do you remembering us? | remembering 应改为 remember，但问谁是否记得我们仍清楚，按语义通过。 |
| Bring me a fork, Tom. | 給我拿把叉子，湯姆。 | ✗ 給我一桶水。 | ✗ 請給湯姆。 | ✗ 給湯姆，湯姆。 | ✗ 给汤姆，湯姆。 | 重复汤姆，叉子没有译出。 |
| 給我拿把叉子，湯姆。 | Bring me a fork, Tom. | ✗ I bought a Tom, but I lost it. | ✗ I have a lot of Tom. | ✗ I got to tell Tom, but I've ever seen you. | ✗ Give me, Tom turned up, I am all the ticket. | 有 give me 和汤姆，仍未说明要叉子，且新增无关信息。 |
| The dog walked across the street. | 這隻狗穿過街道。 | ✗ 這個男孩跳入了水中。 | ✗ 狗喂狗喂狗。 | ✗ 這個老人拄著這個走。 | ✗ 狗喂狗喂狗了下来。 | 有狗，但走过街道变成喂狗的重复片段。 |
| 這隻狗穿過街道。 | The dog walked across the street. | ✗ There's no water in the well. | ✗ This watch is the dog. | ✗ The dog is a doger crank. | ✗ The dog is closely across the dog. | 有 dog 和 across，街道却换成另一只狗，没有表达穿过街道。 |
| Is Tom an actor? | 汤姆是演员吗？ | ✗ 我喝湯姆嗎？ | ✗ 我為得多久？ | ✗ 湯姆嗎？ | ✗ 汤姆是个常吗？ | 保留汤姆和问句，但“常”没有表达演员。 |
| 汤姆是演员吗？ | Is Tom an actor? | ✗ Tom is a teacher? | ✗ Who is the play begin? | ✗ Is Tom a few shoes? | ✗ Is Tom a listen to? | 只有汤姆和问句框架，没有演员。 |
| Can you save this seat for me? | 你可以幫我保留這個座位嗎? | ✗ 你可以给我一起喝。 | ✗ 你能帮我把这译要嗎？ | ✗ 你能給我的嗎？ | ✗ 你能帮我吗？ | 只问能否帮忙，没有说明帮忙留座。 |
| 你可以幫我保留這個座位嗎? | Can you save this seat for me? | ✗ Do you want to me a me? | ✗ Can I can go to the sch? | ✗ Would you save this dress for me? | ✗ Can I help you? | 帮忙的方向反了，也没提座位。 |

<details><summary>本组原句署名</summary>

- CC-BY 2.0 (France) Attribution: tatoeba.org #11291130 (Aphima) & #5113323 (xjjAstrus)
- CC-BY 2.0 (France) Attribution: tatoeba.org #434614 (lukaszpp) & #9583405 (jin1)
- CC-BY 2.0 (France) Attribution: tatoeba.org #5756645 (CM) & #13670883 (LeviHighway)
- CC-BY 2.0 (France) Attribution: tatoeba.org #239100 (CK) & #874518 (Martha)
- CC-BY 2.0 (France) Attribution: tatoeba.org #2649011 (CK) & #8877876 (crescat)
- CC-BY 2.0 (France) Attribution: tatoeba.org #58208 (CK) & #866146 (Martha)

</details>

## 与旧语料相近的两个诊断句对

| 原句 | 参考译文 | 上一版 | 384 句对 | 广泛采样 | 难点采样 | 难点组判定依据 |
| --- | --- | --- | --- | --- | --- | --- |
| This ticket is valid for three days. | 這張票的有效期是三天。 | ✗ 這個杯子裡的水。 | ✗ 這個人都很高。 | ✗ 这家店在牢骚。 | ✗ 这家旅行转帐付款。 | 谈旅行付款，未说三天有效。 |
| 這張票的有效期是三天。 | This ticket is valid for three days. | ✗ This is little water in the pond. | ✗ This is the most important me. | ✗ This plan is a very ticketic afternoon. | ✗ This is going to be a table of this year. | 换成有关桌子和年份的句子。 |
| Who stole the apples? | 誰偷了這些蘋果？ | ✗ 這個池塘裡的水吗？ | ✗ 蘋果汁了吗？ | ✗ 這些蘋果？ | ✗ 谁来了蘋果？ | 有谁和苹果，把偷换成来，不能表达偷苹果。 |
| 誰偷了這些蘋果？ | Who stole the apples? | ✗ Would you like some more expensive than milk? | ✗ How much is your new? | ✗ Who was the apple? | ✗ Who does this apple? | 保留谁和苹果，未表达偷这一动作。 |

<details><summary>本组原句署名</summary>

- CC-BY 2.0 (France) Attribution: tatoeba.org #530682 (papabear) & #905793 (Martha)
- CC-BY 2.0 (France) Attribution: tatoeba.org #1636184 (ari_diary) & #13167227 (LeviHighway)

</details>

