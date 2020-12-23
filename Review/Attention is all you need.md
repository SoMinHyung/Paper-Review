# 「Attention is all you need 」 Review  

![image](https://user-images.githubusercontent.com/11614046/102729530-6d108a80-4374-11eb-81af-74795aa1335c.png) 

https://arxiv.org/abs/1706.03762

NLP 모델에서 자주 등장하는 transformer에 대해 정리해본다. 
이 논문은 NIPS 2017에 등장하여 MT task에서 SOTA를 찍었고, best NLP model로 큰 주목을 받은 BERT의 모듈로도 사용되었다.

<br/>

## 0. Abstract

 현재 대부분의 sequence model은 인코더와 디코더를 포함하는 복잡한 RNN, CNN모델을 기반으로 한다. 최고의 성능을 보이는 모델도 역시 attention을 이용하여 인코더와 디코더를 연결하는 모델이다. 
 
 우리는 Transformer라는 새로운 간단한 네트워크 구조를 제안한다. 트랜스포머는 오직 어텐션만을 사용하며, RNN과 CNN을 사용하지 않는다. 이 모델은 좋은 성능을 보이면서도, 병렬성이 우수하고 학습시간이 단축되는 장점이 있다. 
 
 이 모델은 WMT 2014 영독번역, 영불번역에서 우수한 성능을 보였으며, 다른 task에서도 일반적으로 좋은 성능을 보였다. 또한, 데이터 사이즈가 크던, 한정되어있던 상관없이 좋은 결과를 보였다.
 
<br/>

## 1. Introduction

 RNN, LSTM, GRU와 같은 모델들은 기계번역에서 SOTA 성능을 이뤄냈다. 수많은 노력을 통해 이러한 recurrent 모델과, 인코더-디코더 모델은 성능의 한계를 끌어올려왔다.

 Recurrent모델은 일반적으로 input과 output에서의 token 위치를 따라서 연산을 한다. 이런 모델은 t시점의 input과 t-1시점의 hidden state를 연산하여, t시점의 hidden state를 구한다. 이 구조는 병렬 연산이 불가능하게 만들며, 문장이 길어질수록 메모리 이슈가 발생한다. 최근 연구들은 인수 분해와 조건적 연산(Factorization tricks and conditional computation)을 통해 연산 효율성을 높였다. 하지만, 본질적인 문제는 해결되지 않았다.

 Attention 방식은 input과 output의 거리에 상관없기 때문에 sequence모델이나 번역 모델에서 주요 부분이 되었다. 그러나 지금까지 대부분의 어텐션구조는 RNN과 결합된 형태로 사용되었다.
 
 우리의 Transformer 구조에서는 RNN과 같은 반복 구조는 피하면서, input과 output의 전체 의존성만을 이용한다. 이 구조는 더 많은 병렬화가 가능하게 하며, SOTA를 얻었다.

<br/>

## 2. Background

 Sequential한 연산을 줄이기 위한 연구에서 Extended Neural GPU, ByteNet, ConvS2S와 같은 로직이 만들어졌으며, 이런 로직들은 CNN을 기본 구조로 삼아 input과 output의 hidden representation을 병렬적으로 연산한다. 이러한 모델에서는 input과 output의 거리가 증가할수록 연산량 역시 증가한다. 이로 인해 먼 거리에 있는 단어 간의 의존성을 학습하기 힘들다는 문제점이 발생한다. **Transformer에서는 Attention-weighted position을 통해서 정확도는 희생하여 연산량을 줄여주었다. 떨어진 정확도에 대해서는 Multi-Head Attention을 통해서 극복한다.** 

 **Self-attention**, 다른 말로는 intra-attention은 하나의 sequence내에 있는 다른 토큰들을 연결해주는 어텐션 기법이다. self-attention은 독해, 요약, 문장 이해 등의 다양한 분야에서 성공적으로 사용되고 있다.

 End2End 메모리 네트워크는 RNN+Attention의 기법으로, 단일 언어 QA와 언어 모델링에서 좋은 성능을 보여준다.

 그러나 RNN이나 CNN없이 Attention만을 이용해서 input과 output을 계산하는 방식은 우리들이 최초이다. 다음 부분에서, Transformer와 self-attention에 대해 설명하고 그 장점을 논하겠다.

<br/>

## 3. Model Architecture

 대부분의 성능좋은 neural sequence 모델은 encoder-decoder 구조를 사용한다. 여기서 인코더는 input token (x1, …, xn)을 연속적인 sequence 표현인 Z = (z1, … , zn)으로 변환한다. Z에 대해서, 디코더는 결과인 (y1, … , yn)을 앞에서부터 하나씩 구한다. 각 step들은 자동회귀(auto-regressive)적이기 때문에, 전의 값이 있어야 다음의 값을 구할 수 있다.

 하지만, 트랜스포머는 이러한 encoder-decoder구조를 따르면서, stacked self-attention과 point-wise fully connected layers를 가진다. 아래 사진의 좌측과 우측 절반을 보면 된다.
 
![image](https://user-images.githubusercontent.com/11614046/102755091-46bd1000-43b1-11eb-8413-4e1611eedba6.png)

<br/>

### 3.1. Encoder and Decoder Stacks

* Encoder

 Encoder는 6개의 동일한 레이어가 쌓인 구조이다. 각 레이어는 2개의 sub-layer가 있다. 첫번째는 multi-head self-attention이고, 두번째는 point-wise fully connected feed-forward network(FCFFN)이다. 2개의 sub-layer를 연결하기 위해 layer-normalization을 한다. 이러한 residual connection을 수월하게 하기 위해서 dimension은 512로 고정하였다.

 코드방식으로 표현하면 아래와 같다.
```python 
Stage1_out = Embedding512 + TokenPositionEncoding512 
Stage2_out = layer_normalization(multihead_attention(Stage1_out) + Stage1_out) 
Stage3_out = layer_normalization(FFN(Stage2_out) + Stage2_out) 

out_enc = Stage3_out 
```

<br/>

* Decoder

 Decoder 역시 6개의 동일한 레이어가 쌓인 구조이다. 인코더 레이어에는 2개의 서브레이어가 있지만, 디코더 레이어에는 인코더의 output에 multi-head attention을 하는 서브레이어가 추가되어 총 3개의 서브레이어가 있다. 인코더와 유사하게, 디코더도 각 서브레이어들을 연결하고, layer-normalization을 한다. 디코더에서는 뒷부분을 참조하는 것을 방지하기 위해, self-attention부분을 마스킹하는 방식으로 수정하였다. 이러한 마스킹은 i 시점에서 i 이전의 시점 데이터만을 참조할 수 있도록 만들어준다.

```python
Stage1_out = OutputEmbedding512 + TokenPositionEncoding512

Stage2_Mask = masked_multihead_attention(Stage1_out)
Stage2_Norm1 = layer_normalization(Stage2_Mask) + Stage1_out
#outenc 는 encoder 결과값.
Stage2_Multi = multihead_attention(Stage2_Norm1 + out_enc) +  Stage2_Norm1
Stage2_Norm2 = layer_normalization(Stage2_Multi) + Stage2_Multi

Stage3_FNN = FNN(Stage2_Norm2)
Stage3_Norm = layer_normalization(Stage3_FNN) + Stage2_Norm2

out_dec = Stage3_Norm

```

<br/>

### 3.2. Attention

Attention 함수는 query와 key-value쌍을 토대로 output을 만들어주는 함수이다. (Query, key, value, output은 모두 vector값이다). Output은 value들의 weighted sum으로 계산되며, 이 weight는 key에 대한 query의 compatibility function을 통해 얻어진다.

<br/>

#### 3.2.1 Scaled Dot-Product Attention

 Scaled dot-product attention에는 3개의 input값이 들어간다. Query와 key는 Dk dimension, values는 Dv dimension의 값이다. Query와 모든 Key들과 행렬곱(dot product)을 하고, √Dk로 나눠준다. 그리고 softmax를 취해 나온 확률 값에 value를 곱하면 된다.

 실제 계산할 때, Query, Key, Value를 하나씩 곱해서 하나씩 attention값을 구하는 것이 아니라 행렬 형태로 묶어서 한번에 계산한다. Queries는 Q, Keys는 K, Values는 V로 표현한다. 수식으로 표현하면 다음과 같다.

![image](https://user-images.githubusercontent.com/11614046/102841656-b8946880-4448-11eb-81a9-758adea64142.png)

 수식으로 표현하면 아래와 같다.
 
```python
def attention(Q, K, V):
        num = np.dot(Q, K.T)
        denum = np.sqrt(K.shape[0])
        return np.dot(softmax(num / denum), V)
```

 흔히 사용되는 attention은 additive attention과 dot-product attention이 있다. Dot product attention이 우리가 사용한 알고리즘과 유사하며 다른점은 √dk로 나눠줬다는 것이다. Additive attention은 feed forward network를 통해 양립성을 계산하는 방식이다. Dot-product attention은 최적화된 행렬곱을 이용하기 때문에 빠른 속도와 높은 메모리효율성을 가진다.

 Additive attention은 dk의 값이 작다면 scaling할 필요가 없기 때문에 dot-product보다 성능이 좋다. 하지만, dk가 커진다면 dot-product의 값이 커지기 때문에, softmax를 취했을 때의 미분값이 작아지는 문제점이 있다. 이에 대응하기 위해 우리는 dot-product값을 √dk로 scaling해주었다.



![image](https://user-images.githubusercontent.com/11614046/102757276-6bff4d80-43b4-11eb-93f2-4bb042c01e52.png)

(추가 설명1) Q, K, V가 무엇인지 설명하자면, Q는 attention값을 알고 싶은 단어이다. K와 V는 사실 같은 단어를 의미한다. 하지만 두개로 나눈 이유는 key값을 위한 vector와 value값을 위한 vector를 따로 만들어서 계산하기 위함이다. Self-attention의 경우에는 Q는 토큰, K=V는 하나의 sequence tokens가 될 것이다.

(추가 설명2) 연산과정에 대해서 추가적으로 설명하자면 우리는 query가 어떤 단어와 연관이 있는지 알기 위해 모든 key들과 연산을 한다. Q와 K를 dot-product한 뒤, softmax를 취하면 우리는 하나의 query가 모든 key들과 연관될 확률 값을 구하는 것이다. 따라서, 나온 확률 값과 value를 곱해주면 query가 key(=value)와 연관된 정도를 반영한 벡터값들을 얻을 수 있다.

<br/>

#### 3.2.2 Multi-Head Attention 

![image](https://user-images.githubusercontent.com/11614046/102757423-a5d05400-43b4-11eb-9819-2c44e470fc88.png)

 Multi-head attention은 기존의 attention을 한번만 적용하는 것이 아니라, h번 반복하여 attention을 적용한다. 각 Q, K, V는 linear하게 h번 반복하여 생성하며, attention연산을 병렬적으로 처리한다. 이렇게 구한 값들은 마지막에 다 더해주고, W를 곱해서 dimension사이즈에 맞게 조정해줘서 최종 값을 구한다.

 식으로 표현하면 아래와 같다
 
![image]( https://user-images.githubusercontent.com/11614046/102860450-3ec4a500-4471-11eb-9c97-020dbc66b7a8.png)
 
이 논문에서는 h=8으로 설정했기 때문에, head의 개수는 8개이다. 따라서, 각 vector의 dimension은 8로 나눠진다. Dmodel = 512이기 때문에, dk = dv = 64이다. 전체 과정을 그림으로 표현하면 아래와 같다.

![image](https://user-images.githubusercontent.com/11614046/102860273-d5449680-4470-11eb-98b1-3e43ad342abb.png)

<br/>

#### 3.2.3 Applications of Attention in our model

Transformer는 multi-head attention을 3가지 다른 방식으로 사용한다.

 * Encoder-Decoder attention layer에서 Query는 이전 decoder layer, Key Value는 Encoder에서 온다. 따라서, decoder의 모든 token들은 input sequence의 모든 토큰들을 참조할 수 있다.

 * encoder는 self-attention layer를 가진다. 여기서의 Q, K, V는 이전 layer의 같은 sequence에서 온다. 따라서, encoder는 이전 layer의 모든 위치를 참조할 수 있다.

 * decoder는 self-attention layer에서 모든 위치의 token들을 참조할 수 있지만, 자신의 앞에 위치한 토큰들만 참조하도록 만든다. 즉, sequence내에서 자신의 앞에 위치한 토큰들만 참고할 수 있다. 이는 masking을 통해서 구현했다.

<br/>

### 3.3 Position-wise Feed-Forward Networks

 각각의 Encoder와 Decoder layer 내에는 attention sub-layer말고도, FFN네트워크라는 sub-layer가 존재한다. 이 네트워크는 두 개의 linear transformation으로 구성되어 있으며, 한번의 transformation 후에 ReLu함수를 거쳐 다시 transformation을 한다. 식은 아래와 같다.

![image](https://user-images.githubusercontent.com/11614046/102863117-7b929b00-4475-11eb-8dd9-9e88c6ceb1e0.png)

<br/>

### 3.4 Embeddings and Softmax

다른 sequence 모델과 유사하게, input token과 output token을 D(model)로 바꿔준다. (512차원) 또한, decoder에서 나온 결과에 linear transformation과 softmax를 적용하여, 다음 토큰의 확률값을 계산한다. 우리의 모델에서는 2개의 임베딩 레이어와 softmax이전의 linear transformation에서 동일한 가중치행렬을 사용했다. 임베딩 레이어에서 해당 가중치에는 √D(model)을 곱한다.

<br/>

### 3.5 Positional Encoding

이 모델에서는 RNN이나 CNN을 사용하지 않기 때문에, 추가적으로 토큰의 위치 정보를 제공해야 한다. 이를 위해 토큰의 embedding된 값에 위치 정보를 넣어준다. 위치정보는 다양한 방법으로 구할 수 있으나, 여기서는 sin, cos함수를 이용하여 구한다. 식은 다음과 같다. 

![image](https://user-images.githubusercontent.com/11614046/102868964-45a5e480-447e-11eb-9f6b-687ee6c5b5ba.png)

<br/>

## 4. Why Self-Attention

 이 파트에서는 recurrent나 convolution을 사용하지 않고 self-attention을 사용한 이유에 대해서 알아보겠습니다. self-attention을 사용한 이유는 3가지 장점이 있기 때문이다.

1. 레이어마다 전체 연산량이 줄어든다.

2. 병렬 연산이 가능한 부분이 늘어난다.

3. 먼 거리에 있는 토큰들의 dependency도 잘 계산할 수 있다. 
	- 장거리 dependency를 계산하는 것은 그동안 많은 모델에서 주요 도전과제였다
	- attention방식은 장거리의 dependency를 네트워크의 길이에 상관없이 잘 전달한다.
	
<br/>

![image](https://user-images.githubusercontent.com/11614046/102951350-0031f780-4510-11eb-84da-3e95ddd97c47.png)

**_Self-attention vs recurrent_** 
 
위의 표처럼 self-attention은 1번만에 특정 토큰과 연결될 수 있지만, recurrent모델은 n번이 지나야 특정 토큰과 연결될 수 있다. 연산 복잡성부분에서, n(sequence length)가 d(dimension size)보다 작다면 self-attention은 recurrent모델보다 연산량이 적어진다. 최신 토큰화 방식에서 대부분 dim사이즈가 큰 경향이 있기 때문에, self-attention은 효율적이다. 만약 sequence length가 너무 길어진다면, self-attention(restricted)모델처럼 전체를 attention연산하지않고, r개만큼의 이웃된 토큰들만 연산하도록 한다. 그러면 연산량은 r * n *d로 줄어들지만, 특정 토큰과의 거리가 n/r로 증가하는 단점이 있다.

<br/>

**_Self-attention vs convolution_**

하나의 convolution 레이어에서 k(kernel) < n(sequence length) 라면, 모든 input과 output을 연결하지 못합니다. 그렇다면 n/k개의 cnn 레이어를 쌓거나, logk(n)이라는 희석된 cnn을 사용해야하며, 이는 특정 단어간의 거리가 멀어지게 되는 문제가 발생한다. 따라서, CNN은 RNN보다 비교적으로 비싼 비용을 요구한다. 그러나 분산된 convolution layer는 (k * n * d + n * d^2)으로 복잡성을 줄일 수 있다. k=n인 경우에나 분산된 convolution layer의 복잡성이 self-attention의 복잡성과 같아지기 때문에, 일반적으로 self-attention이 더 효율적이라고 할 수 있다.

<br/>

**_추가적인 self-attention의 장점_**

추가적으로, self-attention은 더 많은 해석가능한 모델을 만들 수 있다. 개별 attention이 어떤 대상들을 주목하고 있는지를 쉽게 알고 해석할 수 있을 뿐만 아니라, multi-head attention이 어떻게 동작하는지 알기 쉽다는 장점이 있다. 


<br/>

## 5. Result 

#### 5.1 Machine Translation

- WMT 2014 영독 번역에서 SOTA달성. 
- WMT 2014 영불 번역에서 SOTA달성. 
- 연산량도 이전의 모델들보다 줄어들었다. 

<br/>

#### 5.2 Model Variation

- single-head의 성능은 낮았으며, multi-head에서도 head의 개수가 너무 많으면 성능이 오히려 저하됨.
- attention dimension 사이즈가 너무 작거나 크면 성능이 저하됨.
- 모델이 커질수록 성능이 개선되며, dropout은 성능을 개선시킴.
- Positional embedding과 sinusoid의 성능은 동일함.

<br/>

#### 5.3 English Contituency Parsing (구성구문분석)

번역 외에도 다른 업무에도 사용될 수 있는지 테스트하기 위해 영어 구문분석을 해봤다. 
WSJ의 40K문장만을 가지고, 특별한 fine-tuning이 없었음에도 RNN기반의 다른 모델보다 더 좋은 성적을 보여주었다.

<br/>

## 6. Conclusion

 이 논문에서는 오로지 attention만을 활용한 Transformer를 제안한다. 이는 지금까지 주로 사용되어왔던 RNN기반의 encoder-decoder구조를 multi-head attention구조로 바꿀 것으로 예상된다.

 번역에서 Transformer는 recurrent, convolution모델보다 더욱 빠르게 학습할 수 있다. WMT 2014 영독, 영불 번역에서는 SOTA를 얻었다. 

 Attention기반의 모델들은 번역뿐만 아니라 다른 분야에도 적용될 수 있을 것이라고 예상된다. 빠른 학습속도 덕분에 Transformer는 text뿐만이 아니라 image, audio, video 등 다양한 input데이터에도 적용할 수 있을 것으로 예상된다.
 

***
**개인적인 감상**

"Neural machine translation by jointly learning to align and translate"에서 제시된 attention은 모든 hidden_state에 대해서 매번 모든 softmax연산을 해야하기 때문에 속도가 느려진다는 점이 있다. (연산량이 꽤 많이 증가한다.)

또한, attention은 기존의 트렌드이던 RNN과 CNN에 추가적인 레이어처럼 사용되어 왔다.

하지만 Trnasformer의 등장으로 attention이 메인이 될 수 있었으며, 한번에 묶어서 계산하기에 연산량 감소와 빠른 속도를 얻을 수 있게 되었다.

덕분에 더욱 더 많은 양의 데이터를 효율적으로 학습시키는 다양한 방법론들이 뒤이어 나올 수 있었다고 생각한다.

## Reference

https://reniew.github.io/43/

https://20chally.tistory.com/223

http://jalammar.github.io/illustrated-transformer/

https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.X9_0SC1ywmm