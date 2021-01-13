# 「Deep Contextualized word Representation 」 Review  

![image](https://user-images.githubusercontent.com/11614046/103964673-5dc87400-519f-11eb-8015-36aa586f0814.png) 

https://arxiv.org/abs/1802.05365

Bi-LSTM을 이용하여, context 정보를 반영하는 pre-training을 제시한 논문이다.  
ELMO는 Ch.3에서 등장하는 Embeddings from Language Models의 약자이다.

<br/>

## 0. Abstract

우리는 (1) 단어의 복잡한 특성(구문과 의미)과 (2) 이들이 어떻게 언어적 문맥에서 사용되는지(다의성)을 모델링하는 새로운 종류의 Deep Contextualized word Representation(문맥과 깊게 연관된 단어표현)을 제안한다. 

우리의 word vectors는 대용량 corpus에서 훈련된 deep bidirectional language model (biLM)의 내부 상태에서 학습한다. 이런 표현(representation)은 다른 모델에 쉽게 적용할 수 있으며, QA TE(textual entailment) 감성분석 등의 6가지 NLP과제에서 SOTA를 달성할 수 있었다.

<br/>

## 1. Introduction

 사전학습된 단어 표현은 많은 NLU 모델에서의 핵심요소이다. 그러나, 고품질의 단어 표현을 얻는 것은 어려운 일이다. 왜냐하면, 이들은 (1)  단어사용의 복잡한 특성(구문과 의미)과 (2) 이들이 어떻게 언어적 문맥에서 사용되는지(다의성)을 이상적으로 반영해야하기 때문이다. 이 논문에서, 우리는 새로운 deep contextualized word representation을 소개하겠다. 이 표현방식은 앞의 어려움을 해결하고, 쉽게 기존의 모델에 사용될수 있어서, 언어 이해와 관련된 문제에서 SOTA를 얻었다.

 우리의 표상은 각 토큰이 각 표현과 1대1로 연결되는 전통적인 모델과는 다르다. 우리는 대용량 코퍼스를 통해 한 쌍의 언어모델을 학습한 BI-LSTM에서 나온 벡터값들을 사용했다. 이러한 이유로 우리는 이를 ELMO(Embeddings form Language MOdels)라고 부른다. 표면적으로 word vector에 접근하는 이전의 방식과는 다르게, ELMO는 biLM의 모든 내적 레이어를 사용한다는 점에서 깊은 표현 방식이다. 더 구체적으로, 우리는 모든 레이어의 값들을 linear combination한 값을 구했으며, 이는 단순히 LSTM의 최종 레이어만을 사용하는 것보다 성능이 좋다.
 
 이런 방식으로 내적 state들을 합쳐주는 것은 매우 풍부한 word representation이 가능하게 한다. 내부 분석을 해보면, 높은 레벨의 LSTM state들은 문맥과 관련된 단어의 의미를 배우고, 낮은 층은 문법적인 측면(POS)을 배우는 경향이 있다. 이 모든 값들을 동시에 보여주는 것은 LM이 각 업무에 맞는 단어 표현을 찾아서 배울 수 있게 하기 때문에 매우 효율적이다.

 추가적인 실험에 따르면 ELMO 표현은 실전에서도 매우 잘 작동하는 것으로 밝혀졌다. ELMO는 QA TE(textual entailment) 감성분석을 포함한 6가지의 분야의 모델에 쉽게 추가될 수 있다. 쉽게 추가될 뿐만 아니라, 각 경우의 성능을 SOTA로 올려주고, 상대적으로 에러율이 20%정도 감소하였다. 직접적으로 비교가 가능한 부분에서 ELMO는 CoVe(기계번역 인코더를 사용한 문맥적표현방식)보다 높은 성능을 보였다. 마지막으로, ELMO와 CoVe를 통해 LSTM의 상부 레이어만을 사용하는 것보다 깊은 표현을 사용하는 것이 성능이 좋다는 사실은 공통적이었다. 우리의 학습 모델과 코드는 공개해놨으며, ELMO가 다른 NLP과제에서 좋은 성능을 제공할 것이라고 예상한다.
  
<br/>

## 2. Related work

 사전에 학습하는 방식은 (word2vec, GloVe 등) 단어의 문법적, 의미적 정보를 반영할 수 있다는 능력때문에 QA, TE, semantic role labelling등의 SOTA NLP모델에 사용되었다. 그러나 이러한 word vector 방식은 한 단어에 하나의 independent representation 만 생성할 수 있다는 단점이 있다. 

 이러한 단점을 극복하기 위해서 이전에 제시된 방법들로는 word vector에 subword information을 추가하거나, 각 단어에 별로 서로 다른 representation을 생성하는 방법들이 있었다. 우리의 방식도 character convolution을 이용하여 혜택을 보며, multi-sense정보를 훈련시키지 않고도 매끄럽게 추가할 수 있다.

 다른 최신 연구들은 문맥의존적(context-dependent)인 표현 모델링에 집중하였다. Context2vec 은 bi-LSTM을 사용해서 중심이 되는 특정한 단어 주변의 문맥을 임베딩한다. 문맥 임베딩의 다른 접근방식은 문맥에 중심 단어까지 포함하고, 지도학습 방식의 인코더나 비지도학습 언어모델을 통해 임베딩하기도 한다. 이러한 접근 방식들은 기계번역 방식이 데이터 병렬 규모의 제한을 받는다는 사실에도 불구하고 대규모 데이터셋으로부터 이득을 봤다. 이 논문에서, 우리는 풍부한 단일언어 데이터를 최대한 이용할 것이며, biLM은 3000만 문장을 학습할 것이다. 또한, 우리는 이런 deep contextual representation 방식이 다양한 NLP과제에 잘 작용한다는 사실을 보여주도록 하겠다.
  
 이전의 연구들은 deep biRNN의 다양한 층들이 서로 다른 특성을 인코딩한다는 것을 보여줬다. 예를 들어, deep LSTM의 낮은 층의 문법적 관리(POS tagging)을 이용하면 dependency parsing, CCG super tagging의 성능이 개선된다. RNN기반의 encoder-decoder 기계번역 시스템에서도, 2개의 LSTM 중 첫번째 레이어를 활용하면 두번째 레이어보다 POS tagging의 성능이 올라간다. 마지막으로 LSTM의 높은 층은 단어문맥적 의미를 인코딩하여 문맥을 반영한다. 비슷한 결과들이 ELMO에서도 유도될 것이고, 기존의 task에 적용된다면 효과가 좋을 것이다.

 Dai and LE와 Ramachandram et. al.은 encoder-decoder 쌍을 LM과 sequence autoencder를 이용해서 사전학습시키고, 특정 작업에 맞도록 finetuning을 한다. 반대로, 우리는 unlabeled data로 biLM을 사전학습시키고, weight를 수정하고, 추가적인 과제 맞춤을 할 것이다. 이는 대용량 데이터를 사용한 biLM의 표현력을 극대화하도록 해줄 것이다. 

<br/>

## 3. ELMo: Embeddings from Language Models

 다른 word embedding과는 다르게, ELMO는 입력된 전체 문장에 대한 함수이다. 이는 Sec3.1처럼 character convolution을 하는 2개의 biLM에서 연산되며, Sec 3.2처럼 internal network state를  선형함수를 통해 계산한다. 이런 방식은 준지도학습이 가능하게 하며, biLM은 대규모 코퍼스를 사용하여 학습된다. (Sec 3.4) 그리고, 다양한 기존의 NLP 구조에 쉽게 추가할 수 있다. (Sec 3.3)

<br/>

### 3.1. Bidirectional language Models

* Forward language model

 N개의 연속된 토큰(t1, t2, …, tN)이 주어졌을 때, 전방언어모델(forward language model)은 (t1, … , tN-1)을 통해 tN이 나올 확률을 계산한다.

 최신 언어모델은 토큰 임베딩이나 문자 CNN을 통해 문맥-독립적인 표현을 계산하고 이를 forward LSTM의 L번째 레이어에 전달한다. 각 LSTM레이어들은 문맥-독립적인 결과를 산출한다. LSTM의 상위 레이어의 최종 결과값은 softmax를 통해 다음의 토큰인 tk+1을 구하게 된다.  
 
 <img src="https://user-images.githubusercontent.com/11614046/104151856-67f9a500-5421-11eb-82a3-fa3c303fbdfe.png" width="40%">  
    
 <br/>  
  
    
* Backward language model

 Backward LM은 Forward LM과  방향이 반대라는 점을 빼면 유사하다. 방향이 반대이기 때문에 뒤의 토큰들을 보고 앞의 토큰을 예측한다.
 
 계산 방식 역시 forward LM과 유사하며, tk를 예측하기 위해 (tk+1, … , tN)이 주어진다.  

 <img src="https://user-images.githubusercontent.com/11614046/104152032-d3dc0d80-5421-11eb-9a7f-8681d84158bc.png" width="40%">  
  
 <br/>  

  
* bi-LM

 biLM은 forward와 backward LM을 합친 것이다. 그리고 두 방향의 log likelihood를 최대화하도록 학습한다.  
 
 Token representation(Θx)과 softmax laye(Θs)에 대한 파라미터는 forward, backward LM 에서 동일하게 사용되지만, LSTM에 대해서는 각각의 다른 파라미터를 사용하게 된다.

 종합적으로, 이러한 구조는 Peter et el.(2017)의 구조와 유사하지만, 부분적으로 같은 weight를 사용한다는 점에서 약간은 다르다. 다음 장에서 biLM 레이어의 선형 결합을 통한 새로운 단어 표현에 대해서 알아보도록 하겠다.

 <img src="https://user-images.githubusercontent.com/11614046/104153668-40f1a200-5426-11eb-8d86-e074882c60d9.png" width="40%">  

<br/>

### 3.2. ELMo

![image](https://user-images.githubusercontent.com/11614046/104155855-63d28500-542b-11eb-9840-398dafd16aa3.png)

 ELMo는 biLM의 중간 층들의 representation 결합이다. 순방향 LSTM (그림 왼쪽 파란색 lstm)과 역방향 LSTM (그림 오른쪽 파란색 lstm)이 각각 생성한 hidden state들을 붙인 벡터값들 (그림 위 초록색)에 softmax를 취해 예측하고자 하는 단어를 맞춘다. 만약 임베딩으로 사용하고자 한다면,  초록색 값들을 그대로 사용한다. 

 그리고 **각 LSTM 레이어에 다르게 가중치(Sj^task)를 두고 더해주게 되는데**, 이 가중치는 softmax레이어에서 정규화한 가중치이다. 이 가중치는 사용자가 하고자 하는 task에 맞춰 학습된다. 즉, 과제에 따라 모델이 ELMO임베딩에 어떤 레이어에 attention을 할지 선택하는 과정이라고 할 수 있다.

 <img src="https://user-images.githubusercontent.com/11614046/104155952-89f82500-542b-11eb-90bb-d2f77ccadd44.png" width="40%">

<br/>


### 3.3 Using biLMs for supervised NLP tasks

 사전 학습된 biLM과 NLP과제의 지도학습 구조가 주어진다면, biLM을 사용하여 해당 과제를 향상시키는 것은 간단하다. 1. biLM을 실행하고, 2. 각 단어들에의 모든 레이어 representation을 기록한다. 그리고, 3. 목표 과제의 모델이 이 representation을 배우도록 선형결합을 한다.

 선형결합을 하는 방법은 다음과 같다. 첫째로, 1. biLM이 없는 지도학습 모델의 가장 하단 레이어를 고안한다. 대부분의 NLP 지도학습 모델은 최하단에 동일한 구조를 사용하곤 하는데, 이는 ELMo가 일관된 방법으로 추가될 수 있도록 해준다. 2. 연속된 토큰(t1, t2, …, tn)이 주어지면, 사전학습된 워드 임베딩과 필요하다면 character-based representation을 사용해서 문맥-독립적인 token representation인 xk를 만든다. 그리고, 3. 모델에서 bi-RNN, CNN, feed forward network등을 이용해서 문맥의존적인 hk representation을 구해낸다.
 
 ELMo를 지도학습 모델에 추가하려면, 1, biLM의 weight를 고정시킨다. 2. ELMO 벡터와 문맥을 고려하지 않은 임베딩 xk를 붙인다. 3. 새로 생성한 문맥을 고려한 단어 임베딩을 task모델의 입력값으로 넣는다. 

 SNLI, SQuAD 등의 task에서 위와 같이 ELMo를 추가하면 성능이 더 향상되었다. 또한, dropout을 적용하는 것보다 정규화하는 것이 성능 향상에 도움을 주는 것을 발견하였다.
 
 <br/>

### 3.4 Embeddings and Softmax

 최종 모델에서는 2개의 biLSTM레이어를 사용했다. 각 레이어는 4096개의 unit과 512차원의 출력값이 나온다. 첫번째와 두번째 레이어는 residual connection으로 연결한다. 문맥을 고려하지 않는 임베딩은 2048개의 charater n-gram convolutional filter를 사용했고, 2개의 highway layer를 추가하였다. 결과적으로 biLM은 입력된 각 토큰에 대해 결과를 포함해서 3개의 representation이 나오게 된다. 대조적으로, 전통적인 워드임베딩 방법은 고정된 단어에 대해서 1개의 representation만이 가능하다.  
 
 1B word benchmark에 대해 10번의 epoch를 학습한 뒤에, forward + backward perplexity는 39.7이었으며, CNN-BIG-LSTM은 30.0이었다. 일반적으로 forward와 backward의 perplexity는 유사하게 나왔으나, 전반적으로 backward의 값이 낮은 모습을 보여줬다.  
  
 한번 사전학습이 되면, biLM은 어떤 task든지 representation을 계산할 수 있다. 몇몇 경우에, biLM을 특정 도메인의 데이터를 통해 finetuning하는 것은 perplexity를 낮추고, 성능을 올리는 결과를 가져왔다. 결과적으로, 미세조정된 biLM을 사용하는 것은 하위 task에 좋은 결과를 낳는다.  
   
<br/>

## 4. Evaluation

 아래의 표에서 보여주다시피, ELMo를 추가하면 6가지 NLP task에서 성능을 높일 수 있었다.
 
 ![image]( https://user-images.githubusercontent.com/11614046/104257188-3388f600-54c0-11eb-8261-a914a3f59963.png)

 
 - Question Answering : SQuAD 데이터셋에서 CoVe를 추가하는 것보다 성능이 향상.
 
 - Text Entailment : Premise가 주어지면 hypothesis가 참인지 확인하는 task
 
 - Semantic role labeling : "Who did What to Whom"을 답하는 task
 
 - Coreference resolution : mention이 동일한 대상을 참조하는지 알아보는 task
 
 - Named Entity Extraction : CoNLL 2003 NER task에서 PER, LOC, ORG, MISC를 구분
 
 - Sentiment Analysis : SST-5. very negative ~ very positive로 구분된 5가지 label에서 선택.
 	
<br/>

## 5. Analysis

 Deep contextual representation을 사용하여, 기존의 다양한 task들의 성능이 향상되는 것을 보여줬다. 이 섹션에서는 특정 부분을 빼거나 값을 변경하여, 해당 부분의 역할을 알아보도록 하겠다.

#### 5.1 Alternate layer weighting schemes

 Regularization parameter인 λ는 성능에서 매우 중요한 역할을 한다. 만약 λ의 값이 1이면 weight function이 레이어들을 단순히 평균값을 구하게 만들고, λ의 값이 매우 작아지면 각 레이어들의 가중치가 각각 다르게 적용된다.

 아래의 표는 기존의 성능과 마지막 층만 사용했을 때의 성능과 λ을 적용하여 모든 레이어의 weight를 사용했을 때의 성능을 비교하여 보여준다. 표에서 보이는 바와 같이 λ의 값이 작은 경우에 서로 다른 레이어의 weight를 학습할 수 있어서 더 좋은 성능을 보여준다. 예외적으로 NER과 같이 소규모 학습데이터를 사용하는 경우에는 λ값의 변화에 따른 성능 변화가 없었다. 

 <img src="https://user-images.githubusercontent.com/11614046/104393934-e9b81280-5588-11eb-99ae-61f92854dd5b.png" width="40%">  

 <br/>

#### 5.2 Where to include ELMo?

 이 논문에서 사용한 모든 task architecture에는 biRNN의 가장 낮은 레이어에만 ELMo를 추가하였다. 하지만 아래의 표에서 보이는바와 같이 특정 task에서는 ELMo를 input과 output 레이어 양쪽에 추가해주는 것이 성능을 추가적으로 더 향상시켜준다는 점을 알 수 있다. SQuAD와 SNLI의 경우, biRNN 이후에 Attention을 사용하기 때문에 output에 ELMo representation을 추가하는 것이 성능을 향상시켰다.  

 <img src="https://user-images.githubusercontent.com/11614046/104395113-5df3b580-558b-11eb-8530-dcfea3dd9af0.png" width="40%">  

 <br/>

#### 5.3 What information is captured by the biLM's representations?

 ELMo를 추가하는 것이 word vector만 있을 때보다 NLP성능이 향상되었기 때문에, biLM의 문맥적 representation은 word vector가 잡아내지 못한 어떤 정보를 갖고 있음에 틀림없다. 직관적으로 biLM은 다의어를 명확화(disambiguation, 다의어의 여러 뜻 중 어떤 의미로 쓰였는지 알아내는 것)에 대한 정보를 갖고 있음에 틀림없다. 아래 표에서 ‘play’라는 단어에 대해서 GloVe는 동사 play만을 알 수 있지만, biLM의 문맥적 표현으로는 동사 play와 명사 play(연극)을 구별해 낸다.  

 <img src="https://user-images.githubusercontent.com/11614046/104395861-00606880-558d-11eb-95b1-ade9c26790a9.png" width="80%">  

 <br/>  

 각 layer에서 어떤 표현을 잡아내는지 비교해보기 위해서, CoVe와 성능을 비교하여 보았다. 

 <img src="https://user-images.githubusercontent.com/11614046/104396422-fee37000-558d-11eb-9f3b-41aa00694efb.png" width="80%">  

 * Word sense diambiguation (단어 의미 명확화) : CoVe보다는 우수하며, 2nd layer에서 더욱 잘 잡아낸다. (표 5)
 
 * POS tagging (품사 태깅) : CoVe보다는 우수하며, 1st layer에서 더욱 잘 잡아낸다. (표 6)  
 
 <br/>

 __Implication for supervised tasks__  
 
 위의 테스트를 통해 biLM의 모든 레이어가 중요한 이유를 알 수 있다. 각 레이어마다 잡아낼 수 있는 문맥적 정보가 다르다.  
 
 <br/>

#### 5.4 Sample efficiency

 ELMo가 있을 때, 학습속도도 빠르고 성능도 좋다. 훨씬 효율적으로 학습하는 것으로 보인다.  

 <img src="https://user-images.githubusercontent.com/11614046/104405525-833eee80-55a0-11eb-9607-b24abec65fa1.png" width="60%">  

 <br/>

#### 5.5 Visualization of learned weights
 
 아래의 그림은 biLM 레이어에의 가중치를 표현한 그림이다. Input layer에서 task모델은, corefenrece와 SQuAD는 첫번째 biLSTM layer를 선호한다. Output layer에서 낮은 레이어에 조금 더 가중치를 두지만 상대적으로 균형잡힌 모습을 보여준다.
 
 <img src="https://user-images.githubusercontent.com/11614046/104405579-a5387100-55a0-11eb-983c-9e4e8ab595ef.png" width="60%">  

 <br/>


## 6. Conclusion

 이 논문에서 우리는 biLM으로부터 고품질의 deep context-dependent representation을 얻는 일반적인 접근법을 소개했으며, ELMo를 다양한 NLP task에 적용했을때 큰 성능향상이 있음을 보여줬다.  biLM의 각 층은 서로 다른 문법 / 문맥적 특징들을 효율적으로 encoding하기 때문에, 모든 층을 이용한 representation이 task의 성능을 향상시킬 수 있었다.

***
**개인적인 감상**

 1. 사전학습된 언어 모델을 활용해 word embedding을 생성하면, 문맥을 이해하는 word embedding을 만들 수 있게 되었다. 이는 다양한 NLP 과제에서 인공신경망으로 만든 word embedding을 통해 큰 성능 향상을 얻을 수 있음을 보여주었다. 이처럼 사전학습된 언어적 지식을 다른 모델에 전이학습시키는 것이 가능하게 되었다.
 
 2. 이 논문에서 주장하는 바에 따르면, 하위 레이어는 문법적 정보를 많이 포함하고 있으며 상위 레이어는 의미적 문맥적 정보를 많이 포함하고 있다고 한다. 이러한 점을 잘 기억한다면, 다양한 NLP과제에서 필요한 정보에 맞게 임베딩값을 추출할 수 있을 것이다.
 
 3. 더 나아가, 인공신경망 언어모델을 사용한다면 word embedding의 성능이 높아지기 때문에 개별적인 NLP task에서 적은 량의 데이터를 가지고도 좋은 성능을 얻을 수 있게 되었다. 이는 전이 학습을 활용한 모델에서 공통적으로 나타나는 장점이며, 지도학습 기반 인공신경망 모형을 통해 데이터 레이블링이라는 노가다를 적게 해도 된다는 결과를 낳았다. 
 
 4. 이전에 언어 모델(Language Model)을 만들면, 별다른 실용적 가치를 갖지 못했다. 왜냐하면 아무리 잘 학습되었다고 해도, 다른 과제에 사용하거나 실용적인 생산물을 만들어내지 못한다는 인식이 강했기 때문이다. 하지만, 이 논문을 통해 사전학습된 언어 모델의 지식을 전이시켜, 다른 task에 사용한다면 목표하고자 하는 성능에 보다 쉽게 도달할 수 있다는 사실을 알게 되었다. 
 
 5. 현재는 attention기반의 bert와 같은 모델로 NLP의 중심주제가 넘어갔지만, ELMo는 word2vec과는 다르게 문맥의 정보를 반영하여 동음이의어와 다의어를 다른 값으로 임베딩할 수 있다는 점에서 의미가 있다고 생각한다.
 

## Reference

https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/20/ELMo-Deep-contextualized-word-representations/

https://misconstructed.tistory.com/42

https://brunch.co.kr/@learning/12