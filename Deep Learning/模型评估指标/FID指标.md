---
title: 一文彻底搞懂AIGC的FID指标
source: https://zhuanlan.zhihu.com/p/706814799
description: FID（Frechet Inception Distance）2017年提出：NIPS2017，论文地址（ https://arxiv.org/abs/1706.08500）FID指标评估生成模型（如生成对抗网络GAN，SD等）生成图像质量的常用指标。它通过比较生成图像和真实图像…
tags:
  - clippings
  - FID
related:
---
## FID（Frechet Inception Distance）

2017年提出：NIPS2017，论文地址（ [arxiv.org/abs/1706.0850](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.08500) ）

## FID指标

评估生成模型（如 [生成对抗网络](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLnlJ_miJDlr7nmipfnvZHnu5wiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.VfaLMhNYAFomVgBjVV9filFtZF4WClEK4_y_vXMmTuI&zhida_source=entity) GAN，SD等）生成图像质量的常用指标。它通过比较生成图像和真实图像的 [特征空间](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E7%A9%BA%E9%97%B4&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLnibnlvoHnqbrpl7QiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.fj_v6Mi90JkOmkxLFLpD09r_77kbMKli8Rvd5MzLCck&zhida_source=entity) 中的分布，来衡量生成图像的质量和多样性。（与传统的均方误差 [MSE](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=MSE&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiJNU0UiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.MqMwdQjWXbThMW92xHcD0axX2C0DY5Ya3nONhgWQXlk&zhida_source=entity) 等指标不同，FID考虑了图像特征的分布，而不仅仅是像素值的差异）

FID 的计算公式如下：

$$
F I D = \left\|\mu_{r} - \mu_{g}\right\|_{2}^{2} + T r \left(\right. \underset{r}{\sum} + \underset{g}{\sum} - 2 \left(\underset{r}{\sum} \underset{g}{\sum} \left.\right)^{1 / 2}\right)
$$

其中：

- $\mu_{r}$ 和 $\underset{r}{\sum}$ 分别是真实图像特征的均值和 [协方差矩阵](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E5%8D%8F%E6%96%B9%E5%B7%AE%E7%9F%A9%E9%98%B5&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLljY_mlrnlt67nn6npmLUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.9x2csxHS9C4zeEUlW-yvY8RBp6oBmODyVGhA1aygRpY&zhida_source=entity) 。
- $\mu_{g}$ 和 $\underset{g}{\sum}$ 分别是生成图像特征的均值和协方差矩阵。
- $\left\|\cdot\right\|_{2}$ 表示欧几里得范数。
- $T r$ 表示 [矩阵的迹](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E7%9F%A9%E9%98%B5%E7%9A%84%E8%BF%B9&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLnn6npmLXnmoTov7kiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.GUbpnNaRGN5AGRWCNX_S3un6R8Qww1B9vc8YuP2QmsA&zhida_source=entity) （trace）。

## FID的物理意义

1. [特征分布](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E5%88%86%E5%B8%83&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLnibnlvoHliIbluIMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.crfiYGYfQ0s3coHDplh7fZSoTNYB72dSWCZW437ZBg8&zhida_source=entity) 匹配：FID计算生成图像和真实图像在特征空间中的均值和协方差矩阵的差异。均值反映了特征的中心位置，协方差矩阵反映了特征的分布。通过比较这些 [统计量](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E7%BB%9F%E8%AE%A1%E9%87%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLnu5_orqHph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.13V519lOSVRZYtofyIS51aSFXcmhlYrjcjDEbtejCto&zhida_source=entity) ，FID可以衡量生成图像和真实图像在特征空间中的分布差异。
2. 质量和多样性：低FID值表示生成图像和真实图像在特征空间中的分布更接近，意味着生成图像在质量和多样性上更接近真实图像。高质量的生成图像不仅需要在视觉上逼真，还需要具有足够的多样性以覆盖真实图像的分布。
3. [矩阵迹](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E7%9F%A9%E9%98%B5%E8%BF%B9&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLnn6npmLXov7kiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.aj5aI_IoGwgXlxrNAd4aw0_jB82HJopw8hogO2ELM0k&zhida_source=entity) ：协方差矩阵的迹反映了特征分布的总方差。通过比较协方差矩阵的迹，FID可以评估生成图像和真实图像在特征空间中的分布范围。

## 公式解读及示例

FID的输入是什么？举例，当前有2张图像，一张为原始图像，一张为模型生成图像，两张图像都经过InceptionV3模型，获取最后一个 [池化层](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E6%B1%A0%E5%8C%96%E5%B1%82&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLmsaDljJblsYIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.oLSqxWnv_OE460SYikS8CCXVCcDwqk604_ChC5mJxzc&zhida_source=entity) （全局空间池化层）的 [激活函数](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLmv4DmtLvlh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.HP_oJ0YKvEkP_yeANBSsQo7iwmvb-6F0W_DA6wk_QEU&zhida_source=entity) 输出值作为图像的特征向量（ [编码向量](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E7%BC%96%E7%A0%81%E5%90%91%E9%87%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLnvJbnoIHlkJHph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.61ouH-rVl0-11yrUB1mFd77lLJEU32YbLH0z4OAg28I&zhida_source=entity) ）。因此，默认每张图像都会变为2048维度的特征向量。

均值 $\mu_{r} 和 \mu_{g}$ 都是2048的维度，均值指的是计算FID的数据集中，各个特征向量的制定位置的平均值；举例

```python
# FID示例，均值部分
import os
import numpy as np

test = np.array([[1,2,3],[4,5,6]])
ur = test.mean(axis=0)

# 输出array([2.5, 3.5, 4.5])
```

**知识点1**

表示 [欧几里得范数](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=2&q=%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E8%8C%83%E6%95%B0&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLmrKflh6Dph4zlvpfojIPmlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.Lzsp-T06v7FKV3E5SWUERlJHQJcUJjy9a9g1udi5vV4&zhida_source=entity) ：也叫 [L2范式](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=L2%E8%8C%83%E5%BC%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiJMMuiMg-W8jyIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI0NTIwMTA3OCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.AVnVJdTsZwJxZFODCEaXaVOZJNumP7lJqxCY53yLVMs&zhida_source=entity) ，是向量长度的一种度量方式。表示向量到原点的距离，对于一个n维向量 ，其2范式定义为

**知识点2**

协方差：Covariance在 [概率论](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E6%A6%82%E7%8E%87%E8%AE%BA&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLmpoLnjoforroiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.JK9PYFaB2xg_i3Dv_T6ux8fDHAwXaI9-ttAw6VOZvss&zhida_source=entity) 和统计学中用于衡量两个变量的总体误差。而方差是协方差的一种特殊情况，即当亮哥变量是相同的情况。

协方差表示的是两个变量的总体的误差，这与只表示一个变量误差的方差不同。如果两个变量的变化趋势一致，也就是说如果其中一个大于自身的期望值，另外一个也大于自身的期望着，那么两个变量之间的协方差就是正值。如果两个变量的变化趋势相反，即其中一个大于自身的期望，另外一个却小于自身的期望值，那么两个变量之间的协方差就是负值。

如何计算协方差？公式比较简单 ，举个更形象化的例子：对于一组样本数据 ，协方差的样本 [估计值](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E4%BC%B0%E8%AE%A1%E5%80%BC&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLkvLDorqHlgLwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.5mqwly-GrE50B3g52HlGdpslgJE2wyrO_U4w1gcHqPE&zhida_source=entity) 为 其中，

**知识点3**

协方差矩阵的定义：是一个方阵，用于描述多维数据集各个维度之间的协方差。举例：对于一个 的 [数据矩阵](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E7%9F%A9%E9%98%B5&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLmlbDmja7nn6npmLUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.LLRYw1dQqQLqtEcBy31tgZM5d8IDkGYg3G5gOfCl9i0&zhida_source=entity) X，其中n是样本数据量，p是特征数据维度，协方差矩阵是一个 的矩阵，其中第i行第j列的元素表示第i个 [特征维度](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E7%BB%B4%E5%BA%A6&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLnibnlvoHnu7TluqYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.pYT-UOxHI5huw8vo6DVf4XdtrKRxTZDNvAlH7a2zrks&zhida_source=entity) 和第j个特征维度的协方差。

**知识点4**

矩阵的迹，在 [线性代数](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLnur_mgKfku6PmlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9._2bGMuLCLggrChEyxw8N-yowwEQcAqOBK3jvSPS2Fpc&zhida_source=entity) 中，一个 × 的矩阵 的 **迹** （或 **迹数** ），是指 的 [主对角线](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E4%B8%BB%E5%AF%B9%E8%A7%92%E7%BA%BF&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLkuLvlr7nop5Lnur8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.8ZmTG4xkcA2kBcB5vcV-OWTjvse4MGap7MAPXCqEPJc&zhida_source=entity) （从左上方至右下方的对角线）上各个元素的总和，一般记作tr⁡( )或Sp⁡( )

## 总结

基于以上的知识点，我们回头看一下FID的公式，根据 [代数恒等式](https://zhida.zhihu.com/search?content_id=245201078&content_type=Article&match_order=1&q=%E4%BB%A3%E6%95%B0%E6%81%92%E7%AD%89%E5%BC%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODUzNzU3OTIsInEiOiLku6PmlbDmgZLnrYnlvI8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDUyMDEwNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.QMLtbXitHr4pq2poK5Ybx_X6qTDRoFs6CEBUfR80vKc&zhida_source=entity) 可知， 是不是很像两个数据集的协方差矩阵的差的平方？再加上矩阵的迹的使用，是不是就变成了两个数据集之间特征维度上的方差的相似性呢？
