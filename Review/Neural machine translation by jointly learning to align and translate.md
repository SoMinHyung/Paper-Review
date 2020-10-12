# 「Neural machine translation by jointly learning to align and translate 」 Review  

![image](https://user-images.githubusercontent.com/11614046/95726605-17549d80-0cb4-11eb-8901-09e4033cb2b3.png) 

https://arxiv.org/abs/1409.0473

<br/>

## 0. Abstract

 통계적 기계번역과는 다르게, 신경망 기계번역은 하나의 신경망을 만들어 최대한의 번역성능을 얻는 것을 목표로 한다. 기존의 Encoder-Decoder 방식은 하나의 문장을 고정된 길이의 벡터로 인코딩한 뒤, 다른 언어의 문장으로 디코딩을 하는 방식이다. 
 
 이 논문에서는 고정된 벡터값을 사용하는 방식이 문장이 길어질수록 병목현상이 발생하여 번역 성능이 급격하게 낮아질 것이라고 판단한다. 이런 문제점에 대한 해결책으로, 중요 단어듦만을 타겟 단어로 지정하여 번역의 성능을 높이고자 한다. 이런 방식을 통해 영불번역에서 SOTA를 성취했으며, 결과를 봤을때도 직관적으로 성능이 개선되었다는 점을 볼 수 있다고 한다.
 
 
<br/>

## 1. Introduction

 기존에는 통계적 방식을 통해서 기계번역을 했으나, 최근들어서는 신경망을 설계하여 기계번역을 하는 방안이 등장했다. 대부분의 신경망 방식은 Encoder-Decoder의 구조이다. 이러한 구조는 하나의 문장을 고정된 길이의 벡터값으로 변환했다가, 이 벡터값을 다른 언어로 decoding하는 방식으로 진행된다. 하지만, **문장의 길이가 길어지면 길어질수록 고정된 길이의 벡터에 모든 내용을 압축해서 담는 것이 불가능해지기 떼문에, 문장의 길이가 길어질수록 성능이 급격하게 낮아지는 문제가 발생한다.**

 따라서, **Align과 Translate를 동시에 하는 모델을 제안한다. 이 모델은 가장 유사한 정보를 가지고 있는 벡터값에 집중하여, 문맥을 번역하도록 한다.**  (여기에서는 soft-alignment를 한다고 말하고 있으나, 이 개념은 우리가 알고 있는 attention과 동일한다. 문장 내의 정보를 전부 확인하여, 어순이 다른 것에 대해서도 학습을 할 수 있게 된다.)

 이 방식은 기존의 방식과는 다르게 고정된 벡터값으로 인코딩하는 것이 아니라, 입력된 문장을 벡터의 순서로 표현하여 이러한 벡터들을 상황에 맞춰 사용하게 된다. 
 
 
<br/>

## 2. Background : Neural Machine Translation

일단 기존의 encoder-decoder모델에 대해서 설명을 하고 넘어가도록 하자.

![image](https://user-images.githubusercontent.com/11614046/95719432-bffdff80-0caa-11eb-9642-bfba1ed805e8.png)

<br/>

위의 그림과 같이 encoder와 decoder는 rnn의 구조를 가지고 있다.  

Encoder에서 특정 단어의 input값(x_t)과 이전 시점의 인코딩 값(h_t-1)을 가지고 해당 시점의 인코딩 값(h_t)를 구한다. 이렇게 hidden_state를 계속 전달하여, S라는 fixed_size vector값을 구한다. 

Decoder에서 특정 단어의 output값(y_t)를 구하기 위해, 이전 시점의 output값(y_t-1)과 해당 시점의 state(s_t)값으로 계산한다. 이렇게 state값을 계속 앞으로 전달하고, 그 이전에 구한 단어를 이용해서 다음 단어를 decode하는 방식이다.

<br/>

## 3. Learning to Align and Translate (Attention based encoder-decoder)

Attention으로 대체된 encoder-decoder구조는 다음과 같다. 
 
![image](https://user-images.githubusercontent.com/11614046/95721238-43b8eb80-0cad-11eb-8455-6b5ecd41d7d0.png)

<br/>

### 3.1. Encoder

인코더은 bi-RNN구조로 이뤄져 있다. 

![image](https://user-images.githubusercontent.com/11614046/95722550-0e150200-0caf-11eb-9edd-4f3844c724e5.png)
 
<br/>

위의 그림과 같이 정방향 rnn으로 fw_h_i들을 만들고, 역방향 rnn으로 bw_h_i를 만든다.

해당 fw_h_i와 bw_h_i를 concat해주면, 해당 input값인 x_i에 대한 hidden_state를 구한 것이다. 

기존의 인코더와는 다르게 S라는 fixed_size vector를 구하지 않는다.

<br/>

### 3.2. Decoder

디코더는 기존의 디코더와 크게 다르지 않다.

기존 디코더는 현재 단어(y_i)를 알기 위해, 이전의 단어(y_i-1) + 이전의 state(s_i-1)를 이용해 현재의 state(s_i)를 구했다. 이렇게 구한 s_i로 y_i를 찾아낸다.

![image](https://user-images.githubusercontent.com/11614046/95724719-a8764500-0cb1-11eb-8dd2-1e6f9e6b8bec.png)

<br/>

attention의 디코더 역시 현재의 단어(y_i)를 알기 위해, 이전의 단어 (y_i-1) + 이전의 state(s_i-1)을 이용하는데 추가적으로 encoder에서 만든 hidden_state값을 이용한다.

이는 식으로 나타내면 y_i <- s_i = g(y_i-1, s_i, c)로 표현된다.

우리의 목적은 s_i를 구하는 것이기 때문에, c값을 구하는 방법을 알아보겠다.

c는 s_i-1과 c값을 구성한 h_j값들의 유사도를 계산한 스코어 값이다. 

이렇게 구한 각 h_j들의 스코어값들에 softmax를 취해주면, 해당 h_j의 등장확률로 치환할 수 있다.

높은 확률을 가진 hidden_state들을 강하게 반영해주면, 해당 단어의 가중치가 증가하여 문맥의 흐름을 반영한 디코딩이 가능하다.

<br/>

## 4. Experiment Settings

Attention 알고리즘을 사용하여, 해당 논문은 SOTA결과를 얻을 수 있었다. 

하지만 데이터셋과 결과가 중요한 것은 아니기에 넘어가도록 한다. 앞에서 attention의 등장 배경과 그 내용에 대해서 알아봤으니 충분하다고 생각한다.

주목할만한 것은 논문에서 목표한 바와 같이 아래 그래프처럼 문장의 길이가 길어져도 성능이 줄어들지 않았다는 점이다.

![image](https://user-images.githubusercontent.com/11614046/95726099-76fe7900-0cb3-11eb-9c69-bc6f1821ac8f.png)

<br/>

## 5. Result 

각 단어들의 hidden_state를 가중치로 변환해준 결과이다. 

자신에 해당되는 부분의 가중치가 가장 높았으며, 그 주변의 단어들이 높은 것을 볼 수 있다.

![image](https://user-images.githubusercontent.com/11614046/95726260-a7deae00-0cb3-11eb-8603-8afb4087ada6.png)


## 6. 요약

Attention은 특정 대상에 대해 가중치를 주도록 하는 매커니즘이다.

가중치를 주는 방식은 decoder에서 이전 시점의 state값과 encoder에서 만든 hidden_state의 유사도를 계산하여 도출한다.

attention의 단점으로는 모든 hidden_state에 대해서 매번 모든 softmax연산을 해야하기 때문에 속도가 느려진다는 점이 있다. (연산량이 꽤 많이 증가한다.)

하지만, 이 방법론의 등장으로 bert, gpt와 같은 획기적인 기법들이 등장할 수 있었다.