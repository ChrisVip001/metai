# 125M 翻译逐题记录

同一题的各列均来自该权重的自由生成。✓ 表示整句保留原意，✗ 表示至少一项关键含义出错。[所有对照、原始空白、token ID 和理由](translation-v3.json) · [评测方法](../TRANSLATION.zh-CN.md)

## 未见动作与对象的组合

| 题号与原句 | 参考译文 | 上轮 SFT | 交叉＋肯否成对 | 再扩自然句 | 末列判定依据 |
| --- | --- | --- | --- | --- | --- |
| cube-01<br>Nina likes juice. | 妮娜喜欢果汁。 | ✗ 沒有錢。 | ✓ 妮娜喜欢果汁。 | ✓ 妮娜喜欢果汁。 | 主语、动作、对象和肯否均保留。 |
| cube-02<br>妮娜喜欢果汁。 | Nina likes juice. | ✗ There's no money in the room. | ✓ Nina likes juice. | ✓ Nina likes juice. | 主语、动作、对象和肯否均保留。 |
| cube-03<br>Nina does not like juice. | 妮娜不喜欢果汁。 | ✗ 沒有用卡的。 | ✓ 妮娜不喜欢果汁。 | ✓ 妮娜不喜欢果汁。 | 主语、动作、对象和肯否均保留。 |
| cube-04<br>妮娜不喜欢果汁。 | Nina does not like juice. | ✗ There's no money in the room. | ✓ Nina does not like juice. | ✓ Nina does not like juice. | 主语、动作、对象和肯否均保留。 |
| cube-05<br>Nina likes water. | 妮娜喜欢水。 | ✗ 沒有錢。 | ✓ 妮娜喜欢水。 | ✓ 妮娜喜欢水。 | 主语、动作、对象和肯否均保留。 |
| cube-06<br>妮娜喜欢水。 | Nina likes water. | ✗ There's no money. | ✓ Nina likes water. | ✓ Nina likes water. | 主语、动作、对象和肯否均保留。 |
| cube-07<br>Nina does not like water. | 妮娜不喜欢水。 | ✗ 沒有想要。 | ✓ 妮娜不喜欢水。 | ✓ 妮娜不喜欢水。 | 主语、动作、对象和肯否均保留。 |
| cube-08<br>妮娜不喜欢水。 | Nina does not like water. | ✗ There's no money in the room. | ✓ Nina does not like water. | ✓ Nina does not like water. | 主语、动作、对象和肯否均保留。 |
| cube-09<br>Nina wants juice. | 妮娜想要果汁。 | ✗ 沒有錢。 | ✓ 妮娜想要果汁。 | ✓ 妮娜想要果汁。 | 主语、动作、对象和肯否均保留。 |
| cube-10<br>妮娜想要果汁。 | Nina wants juice. | ✗ There's no need to do whatever you. | ✓ Nina wants juice. | ✓ Nina wants juice. | 主语、动作、对象和肯否均保留。 |
| cube-11<br>Nina does not want juice. | 妮娜不想要果汁。 | ✗ 沒有錢。 | ✓ 妮娜不想要果汁。 | ✓ 妮娜不想要果汁。 | 主语、动作、对象和肯否均保留。 |
| cube-12<br>妮娜不想要果汁。 | Nina does not want juice. | ✗ There's no money in the room. | ✓ Nina does not want juice. | ✓ Nina does not want juice. | 主语、动作、对象和肯否均保留。 |
| cube-13<br>Nina wants water. | 妮娜想要水。 | ✗ 沒有錢。 | ✓ 妮娜想要水。 | ✓ 妮娜想要水。 | 主语、动作、对象和肯否均保留。 |
| cube-14<br>妮娜想要水。 | Nina wants water. | ✗ There's no need to do it. | ✓ Nina wants water. | ✓ Nina wants water. | 主语、动作、对象和肯否均保留。 |
| cube-15<br>Nina does not want water. | 妮娜不想要水。 | ✗ 沒有錢。 | ✓ 妮娜不想要水。 | ✓ 妮娜不想要水。 | 主语、动作、对象和肯否均保留。 |
| cube-16<br>妮娜不想要水。 | Nina does not want water. | ✗ There's no money in the room. | ✓ Nina does not want water. | ✓ Nina does not want water. | 主语、动作、对象和肯否均保留。 |
| cube-17<br>Leo drinks coffee. | 利奥喝咖啡。 | ✗ 我們的時須錢。 | ✓ 利奥喝咖啡。 | ✓ 利奥喝咖啡。 | 主语、动作、对象和肯否均保留。 |
| cube-18<br>利奥喝咖啡。 | Leo drinks coffee. | ✗ She was a lot of the room. | ✓ Leo drinks coffee. | ✓ Leo drinks coffee. | 主语、动作、对象和肯否均保留。 |
| cube-19<br>Leo does not drink coffee. | 利奥不喝咖啡。 | ✗ 沒有須的。 | ✓ 利奥不喝咖啡。 | ✓ 利奥不喝咖啡。 | 主语、动作、对象和肯否均保留。 |
| cube-20<br>利奥不喝咖啡。 | Leo does not drink coffee. | ✗ She was a lot of the room. | ✓ Leo does not drink coffee. | ✓ Leo does not drink coffee. | 主语、动作、对象和肯否均保留。 |
| cube-21<br>Leo drinks tea. | 利奥喝茶。 | ✗ 我們的話。 | ✓ 利奥喝茶。 | ✓ 利奥喝茶。 | 主语、动作、对象和肯否均保留。 |
| cube-22<br>利奥喝茶。 | Leo drinks tea. | ✗ She was a lot of the room. | ✓ Leo drinks tea. | ✓ Leo drinks tea. | 主语、动作、对象和肯否均保留。 |
| cube-23<br>Leo does not drink tea. | 利奥不喝茶。 | ✗ 沒有錢。 | ✓ 利奥不喝茶。 | ✓ 利奥不喝茶。 | 主语、动作、对象和肯否均保留。 |
| cube-24<br>利奥不喝茶。 | Leo does not drink tea. | ✗ She was a lot of the room. | ✓ Leo does not drink tea. | ✓ Leo does not drink tea. | 主语、动作、对象和肯否均保留。 |
| cube-25<br>Leo orders coffee. | 利奥点咖啡。 | ✗ 我們的時間。 | ✓ 利奥点咖啡。 | ✓ 利奥点咖啡。 | 主语、动作、对象和肯否均保留。 |
| cube-26<br>利奥点咖啡。 | Leo orders coffee. | ✗ She was a lot of the room. | ✓ Leo orders coffee. | ✓ Leo orders coffee. | 主语、动作、对象和肯否均保留。 |
| cube-27<br>Leo does not order coffee. | 利奥不点咖啡。 | ✗ 沒有須的。 | ✓ 利奥不点咖啡。 | ✓ 利奥不点咖啡。 | 主语、动作、对象和肯否均保留。 |
| cube-28<br>利奥不点咖啡。 | Leo does not order coffee. | ✗ She was a lot of the room. | ✓ Leo does not order coffee. | ✓ Leo does not order coffee. | 主语、动作、对象和肯否均保留。 |
| cube-29<br>Leo orders tea. | 利奥点茶。 | ✗ 我們的時間。 | ✗ 利奥点啤酒。 | ✓ 利奥点茶。 | 主语、动作、对象和肯否均保留。 |
| cube-30<br>利奥点茶。 | Leo orders tea. | ✗ She was a lot of the room. | ✓ Leo orders tea. | ✓ Leo orders tea. | 主语、动作、对象和肯否均保留。 |
| cube-31<br>Leo does not order tea. | 利奥不点茶。 | ✗ 沒有錢。 | ✗ 利奥不点啤酒。 | ✓ 利奥不点茶。 | 主语、动作、对象和肯否均保留。 |
| cube-32<br>利奥不点茶。 | Leo does not order tea. | ✗ She was a lot of the room. | ✓ Leo does not order tea. | ✓ Leo does not order tea. | 主语、动作、对象和肯否均保留。 |

## 真实饮品句

| 题号与原句 | 参考译文 | 上轮 SFT | 交叉＋肯否成对 | 再扩自然句 | 末列判定依据 |
| --- | --- | --- | --- | --- | --- |
| drinks-33<br>I'm drinking milk. | 我在喝牛奶。 | ✗ 我很抱歉。 | ✗ 我喝牛奶。 | ✓ 我正在喝牛奶。 | “正在”与进行时等义，喝、牛奶和第一人称均正确。 |
| drinks-34<br>你喝啤酒吗？ | Do you drink beer? | ✗ Would you like to do this? | ✗ Are you single? | ✓ Do you drink beer? | 与参考译文一致。 |
| drinks-35<br>I can't drink milk. | 我不可以喝牛奶。 | ✗ 我不能让我们不能的。 | ✗ 我不喝牛奶。 | ✗ 我不喝牛奶。 | 不喝牛奶只陈述行为，原句说的是不能喝。 |
| drinks-36<br>我的朋友都不喝咖啡。 | None of my friends drink coffee. | ✗ My father is a lot of my father. | ✗ I do not drink coffee. | ✗ My baby began crying. | 把朋友都不喝咖啡改成我的宝宝开始哭。 |
| drinks-37<br>Did they buy the juice? | 他們有買果汁嗎？ | ✗ 他們的試著一個嗎？ | ✗ 他們在多雨地區種植水稻。 | ✗ 他买果汁。 | 买果汁译对了，把“他们买了吗”改成“他买”，人数、问句和过去事件均有损失。 |
| drinks-38<br>我要一杯咖啡。 | I want to have a cup of coffee. | ✗ I don't have a lot. | ✗ I do not want coffee. | ✗ I want coffee. | 想要咖啡的意图正确，遗漏了原文明确要求的一杯。 |
| drinks-39<br>How about another cup of coffee? | 要不要再來一杯咖啡？ | ✗ 他們的時候讀要多久? | ✗ 考得如何？ | ✗ 啤酒一瓶多少錢 ? | 把再来一杯咖啡的提议译成询问一瓶啤酒的价格。 |
| drinks-40<br>我要的不是茶，而是咖啡。 | What I want isn't tea, but coffee. | ✗ It's not a lot of the room. | ✗ I do not want coffee. | ✗ I want tea to coffee. | 两种饮品都出现，但没有表达排除茶、想要咖啡的否定范围和对比。 |

<details><summary>Tatoeba 逐题署名</summary>

- drinks-33 · CC-BY 2.0 (France) Attribution: tatoeba.org #3132269 (CK) & #3140495 (GlossaMatik)
- drinks-34 · CC-BY 2.0 (France) Attribution: tatoeba.org #1761849 (CK) & #4264903 (notabene)
- drinks-35 · CC-BY 2.0 (France) Attribution: tatoeba.org #1084522 (CK) & #2147732 (egg0073)
- drinks-36 · CC-BY 2.0 (France) Attribution: tatoeba.org #252024 (CK) & #862713 (Martha)
- drinks-37 · CC-BY 2.0 (France) Attribution: tatoeba.org #2239680 (AlanF_US) & #5548510 (egg0073)
- drinks-38 · CC-BY 2.0 (France) Attribution: tatoeba.org #253436 (CK) & #2302310 (egg0073)
- drinks-39 · CC-BY 2.0 (France) Attribution: tatoeba.org #62229 (CK) & #846454 (Martha)
- drinks-40 · CC-BY 2.0 (France) Attribution: tatoeba.org #1615236 (CK) & #1783780 (sadhen)

</details>

## 真实日常句

| 题号与原句 | 参考译文 | 上轮 SFT | 交叉＋肯否成对 | 再扩自然句 | 末列判定依据 |
| --- | --- | --- | --- | --- | --- |
| everyday-41<br>He translated the verse into English. | 他把诗翻译成英语。 | ✗ 他們的話多人。 | ✗ 他周遊全國各地。 | ✗ 他用水装满了瓶子。 | 换成用水装瓶，漏掉翻译、诗和英语。 |
| everyday-42<br>我不認識這裡的任何人。 | I don't know anybody here. | ✗ I don't know what you want to do. | ✗ I do not order juice. | ✗ I don't think it'll be that easy. | 把不认识这里任何人改成认为某事没那么容易。 |
| everyday-43<br>I bought a dozen pencils today. | 我今天買了一打鉛筆。 | ✗ 我請這個人個人個人。 | ✗ 我喝醉了，做不到。 | ✗ 我买了一支钢笔，但我把它丢了。 | 买的动作正确，一打铅笔被换成一支钢笔，漏掉今天，还编入丢失的情节。 |
| everyday-44<br>這次別迷路。 | Try not to get lost this time. | ✗ There's no money. | ✗ The students are all at their desks now. | ✗ There's no water in the. | 输出没有水的残句，未表达这次避免迷路的提醒。 |
| everyday-45<br>Turn off the gas. | 把煤气关了！ | ✗ 那個男孩把這個人。 | ✗ Tom 最喜欢咖啡。 | ✗ 這個杯子裝滿了。 | 译成杯子满了，遗漏关掉煤气的命令。 |
| everyday-46<br>我得卖我的房子。 | I have to sell my house. | ✗ I'm sorrying to me. | ✗ I sell water. | ✗ I sell my mother. | 认出了卖，把自己的房子错译成母亲，遗漏“得”。 |
| everyday-47<br>Tom doesn't need to worry about it. | 汤姆不需要担心。 | ✗ 汤姆不喜欢足球。 | ✗ 汤姆不买茶。 | ✗ 汤姆不會去學校。 | 保留汤姆和否定，却把不必担心换成不会去学校。 |
| everyday-48<br>她把脏盘子放进水池里。 | She put the dirty dishes in the sink. | ✗ She was a lot of the room. | ✗ She does not order water. | ✗ She was only frightened, not injured. | 把盘子放入水池改成受惊但未受伤，事件不同。 |

<details><summary>Tatoeba 逐题署名</summary>

- everyday-41 · CC-BY 2.0 (France) Attribution: tatoeba.org #1315801 (CK) & #10021369 (GlossaMatik)
- everyday-42 · CC-BY 2.0 (France) Attribution: tatoeba.org #3301001 (CK) & #6860671 (verdastelo9604)
- everyday-43 · CC-BY 2.0 (France) Attribution: tatoeba.org #25765 (CK) & #848978 (Martha)
- everyday-44 · CC-BY 2.0 (France) Attribution: tatoeba.org #8812308 (CK) & #9533739 (verdastelo9604)
- everyday-45 · CC-BY 2.0 (France) Attribution: tatoeba.org #63744 (CK) & #1411609 (asosan)
- everyday-46 · CC-BY 2.0 (France) Attribution: tatoeba.org #1744777 (Amastan) & #5581729 (verdastelo9604)
- everyday-47 · CC-BY 2.0 (France) Attribution: tatoeba.org #4664279 (CK) & #9956878 (GlossaMatik)
- everyday-48 · CC-BY 2.0 (France) Attribution: tatoeba.org #312724 (CK) & #866752 (kooler)

</details>

## 重答训练过的 12 个句对

| 题号与原句 | 参考译文 | 上轮 SFT | 记忆训练后 | 末列判定依据 |
| --- | --- | --- | --- | --- |
| memory-01<br>You look very happy this morning. | 你今天上午看起來很高興。 | ✗ 你應該要這個人。 | ✓ 你今天上午看起來很高興。 | 与参考译文一致。 |
| memory-02<br>你今天上午看起來很高興。 | You look very happy this morning. | ✗ There's no intent to do it. | ✓ You look very happy this morning. | 与参考译文一致。 |
| memory-03<br>That is mine. | 那是我的。 | ✗ 那個人個果。 | ✓ 那是我的。 | 与参考译文一致。 |
| memory-04<br>那是我的。 | That is mine. | ✗ My father is a lot. | ✓ That is mine. | 与参考译文一致。 |
| memory-05<br>I'll get started right away. | 我會立刻開始。 | ✗ 我會再訴會來。 | ✓ 我會立刻開始。 | 与参考译文一致。 |
| memory-06<br>我會立刻開始。 | I'll get started right away. | ✗ I'm sorry to you. | ✓ I'll get started right away. | 与参考译文一致。 |
| memory-07<br>Is Tom able to speak French? | Tom能说法文吗？ | ✗ 汤姆是說的嗎？ | ✓ Tom能说法文吗？ | 与参考译文一致。 |
| memory-08<br>Tom能说法文吗？ | Is Tom able to speak French? | ✗ Would you want to do this? | ✓ Is Tom able to speak French? | 与参考译文一致。 |
| memory-09<br>She must be over eighty. | 她肯定超过80岁了。 | ✗ 她会议我。 | ✓ 她肯定超过80岁了。 | 与参考译文一致。 |
| memory-10<br>她肯定超过80岁了。 | She must be over eighty. | ✗ She was a lot of her. | ✓ She must be over eighty. | 与参考译文一致。 |
| memory-11<br>Think before you act. | 三思而後行！ | ✗ 你應該要什麼。 | ✓ 三思而後行！ | 与参考译文一致。 |
| memory-12<br>三思而後行！ | Think before you act. | ✗ There's no money in the room. | ✓ Think before you act. | 与参考译文一致。 |
| memory-13<br>My room is just below. | 我的房間就在樓下。 | ✗ 我的房间里是很多。 | ✓ 我的房間就在樓下。 | 与参考译文一致。 |
| memory-14<br>我的房間就在樓下。 | My room is just below. | ✗ My father is a lot of my father. | ✓ My room is just below. | 与参考译文一致。 |
| memory-15<br>You are selling him short. | 你別小看了他。 | ✗ 你應該要的。 | ✓ 你別小看了他。 | 与参考译文一致。 |
| memory-16<br>你別小看了他。 | You are selling him short. | ✗ You may go to you. | ✓ You are selling him short. | 与参考译文一致。 |
| memory-17<br>What happened here? | 这儿发生了什么事？ | ✗ 這個這個？ | ✓ 这儿发生了什么事？ | 与参考译文一致。 |
| memory-18<br>这儿发生了什么事？ | What happened here? | ✗ Would you like to do this? | ✓ What happened here? | 与参考译文一致。 |
| memory-19<br>He is a friendly person. | 他是一個友善的人。 | ✗ 他很抱歉。 | ✓ 他是一個友善的人。 | 与参考译文一致。 |
| memory-20<br>他是一個友善的人。 | He is a friendly person. | ✗ He is a lot of the room. | ✓ He is a friendly person. | 与参考译文一致。 |
| memory-21<br>It's nonsense to try that. | 这种尝试很荒谬。 | ✗ 這個人個人個人個人。 | ✓ 这种尝试很荒谬。 | 与参考译文一致。 |
| memory-22<br>这种尝试很荒谬。 | It's nonsense to try that. | ✗ There's no money. | ✓ It's nonsense to try that. | 与参考译文一致。 |
| memory-23<br>I think I'm going to cry. | 我可要哭了。 | ✗ 我能想去。 | ✓ 我可要哭了。 | 与参考译文一致。 |
| memory-24<br>我可要哭了。 | I think I'm going to cry. | ✗ You may be so. | ✓ I think I'm going to cry. | 与参考译文一致。 |

<details><summary>Tatoeba 逐题署名</summary>

- memory-01 · CC-BY 2.0 (France) Attribution: tatoeba.org #242183 (CK) & #871207 (Martha)
- memory-02 · CC-BY 2.0 (France) Attribution: tatoeba.org #242183 (CK) & #871207 (Martha)
- memory-03 · CC-BY 2.0 (France) Attribution: tatoeba.org #67126 (CK) & #6728904 (verdastelo9604)
- memory-04 · CC-BY 2.0 (France) Attribution: tatoeba.org #67126 (CK) & #6728904 (verdastelo9604)
- memory-05 · CC-BY 2.0 (France) Attribution: tatoeba.org #2543673 (CK) & #6119482 (verdastelo9604)
- memory-06 · CC-BY 2.0 (France) Attribution: tatoeba.org #2543673 (CK) & #6119482 (verdastelo9604)
- memory-07 · CC-BY 2.0 (France) Attribution: tatoeba.org #2953015 (CK) & #4844993 (pig8322)
- memory-08 · CC-BY 2.0 (France) Attribution: tatoeba.org #2953015 (CK) & #4844993 (pig8322)
- memory-09 · CC-BY 2.0 (France) Attribution: tatoeba.org #310212 (CK) & #334099 (fucongcong)
- memory-10 · CC-BY 2.0 (France) Attribution: tatoeba.org #310212 (CK) & #334099 (fucongcong)
- memory-11 · CC-BY 2.0 (France) Attribution: tatoeba.org #4495480 (CK) & #873444 (Martha)
- memory-12 · CC-BY 2.0 (France) Attribution: tatoeba.org #4495480 (CK) & #873444 (Martha)
- memory-13 · CC-BY 2.0 (France) Attribution: tatoeba.org #251777 (U2FS) & #788810 (Martha)
- memory-14 · CC-BY 2.0 (France) Attribution: tatoeba.org #251777 (U2FS) & #788810 (Martha)
- memory-15 · CC-BY 2.0 (France) Attribution: tatoeba.org #15799 (CK) & #795334 (Martha)
- memory-16 · CC-BY 2.0 (France) Attribution: tatoeba.org #15799 (CK) & #795334 (Martha)
- memory-17 · CC-BY 2.0 (France) Attribution: tatoeba.org #51011 (fucongcong) & #344411 (fucongcong)
- memory-18 · CC-BY 2.0 (France) Attribution: tatoeba.org #51011 (fucongcong) & #344411 (fucongcong)
- memory-19 · CC-BY 2.0 (France) Attribution: tatoeba.org #302443 (CK) & #833216 (Martha)
- memory-20 · CC-BY 2.0 (France) Attribution: tatoeba.org #302443 (CK) & #833216 (Martha)
- memory-21 · CC-BY 2.0 (France) Attribution: tatoeba.org #41627 (CK) & #335012 (fucongcong)
- memory-22 · CC-BY 2.0 (France) Attribution: tatoeba.org #41627 (CK) & #335012 (fucongcong)
- memory-23 · CC-BY 2.0 (France) Attribution: tatoeba.org #2407900 (CK) & #6044371 (verdastelo9604)
- memory-24 · CC-BY 2.0 (France) Attribution: tatoeba.org #2407900 (CK) & #6044371 (verdastelo9604)

</details>

