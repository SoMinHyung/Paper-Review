# 「Improving Language Understanding by Generative Pre-Training(GPT-1) 」 Review  

![image](https://user-images.githubusercontent.com/11614046/105116360-f818a800-5b0d-11eb-948e-e0a0659ae5d7.png) 

https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

NLP에서 BERT와 함께 양대산맥이라고 할 수 있는 GPT에 대해서 알아볼 계획이다.  
Encoder와 Decoder를 사용하는 attention기반의 Transformer가 2017년애 발표되었고, Decoder를 사용하는 GPT-1의 논문 2018년 6월에 발표되었다. (Encoder를 사용하는 BERT는 18년 10월)

<br/>


## 0. Abstract

 자연어이해(Natural Language Understanding, NLU)에는 원문함의(textual entailment, TE), 질의응답(QA), 문장 유사도 평가, 문서 분류 등 다양한 범위의 task가 존재한다. 라벨링되지 않은 텍스트 데이터는 엄청나게 많지만, 이러한 특정 task에 사용할 라벨링된 데이터는 희소하기 때문에 각 task에 적합한 개별적인 모델을 만드는 것이 어렵다. 따라서, 라벨링되지 않은 대규모 텍스트 데이터를 사용하여 생성적 사전학습(Generative Pre-training)을 한 LM(Language Model)을 만든 뒤, 각 task에 fine-tuning을 한다면 이러한 과제들애의 큰 성능향상이 가능하다. 이전의 접근 방식과는 다르게, 우리는 fine-tuning을 할 때 모델 구조는 최소한으로 변하면서 효율적인 전이학습이 가능하도록 과제에 맞는 입력표현(input transformation)을 이용했다. 우리의 접근법이 NLU에 있어서 다양한 벤치마크에서 효율적이라는 것을 보여주겠다.  

 우리의 과제를 모르는(task-agnostic) 모델은 개별적인 과제에 특화된 모델들보다  성능이 앞섰고, 조사된 12개의 과제 중에서 9개에서 SOTA를 달성했다. 예를 들어, 상식추론(commonsense reasoning)에서 8.9%, QA에서 5.7%, TE에서 1.5%의 성능향상을 보였다.

<br/>


## 1. Introduction

 원문으로부터 효율적으로 배울 수 있는 능력은 NLP에서 지도학습에의 의존성을 줄이기 위해 필수적이다. 대부분의 딥러닝 방법론들은 상당한 양의 수동-라벨링된 데이터를 기반으로 특정 과제를 목표로 하기 때문에, 다양한 과제에 범용적으로 사용될 수 있는 모델의 개발이 어렵다. 이러한 상황에서 라벨링되지 않은 데이터로 부터 언어적 정보를 얻어낼 수 있는 모델은 만들기 힘들고 비싼 라벨링 데이터를 얻을 대안이 된다. 추가적으로 충분한 지도가 있다면, 비지도 방식의 표현(representation)을 얻는 것이 성능적으로 더 좋은 결과를 낳을 수도 있다. 다양한 NLP과제에서 사전학습된 word embedding을 사용하여 성능을 높인 것이 그 예이다.  
 
 그러나 unlabeled 텍스트에서 word-level 이상의 정보를 얻는 것은 2가지 이유로 어렵다. 첫째, 어떤 종료의 최적화 함수(Optimization objectives)가 전이에 유용한 text representation을 배우는지 모른다. 최근 연구에 따르면 언어 모델링, 기계 번역, 담화 일관성(discourse coherence) 등 다양한 분야에서 각각 다른 방법이 성능이 좋았다. 둘째, 이렇게 학습된 것을 다른 과제로 어떻게 전이해야 가장 효과적인지 일치된 의견이 없다. 현재 기술들은 특정 과제에 맞는 모델 구조를 설계하고, 복잡한 학습 계획과 보조적인 학습 목표를 추가하는 방식이다. 이러한 불확실성은 언어 처리에 있어 효율적인 준지도학습 접근이 어렵게 만드는 요소이다.  
 
 이 논문에서 우리는 비지도학습을 통한 사전학습과 지도학습을 통한 세부조정을 둘 다 이용해서 언어 이해를 위한 준지도 학습 접근법에 대해 알아볼 것이다. 이 논문의 목표는 약간의 변형만으로도 다양한 과제에 전이시킬 수 있는 범용적인 표현(representation)을 학습하는 것이다. 우리의 가정은 대용량의 unlabeled 텍스트와 수동으로 labeling한 데이터를 갖고 있는 dataset이 존재한다는 것이다. 목표 과제가 unlabeled 텍스트와 동일한 주제일 필요는 없다. 학습 과정은 두 단계를 거친다. 첫째, 신경망모델의 초기 parameter를 학습하기 위해, unlabeled 데이터에 대한 언어 모델링 목적함수를 사용한다. 둘째, 이 파라미터를 지도 목적함수를 이용하여 목표 과제에 적용한다.  
 
 우리의 모델 구조로는 기계 번역, 문서 생성, 구문 분석과 같은 다양한 과제에서 강력한 성능을 보여준 Transformer를 사용한다. 이 모델은 RNN보다 장기 기억력이 뛰어나며, 구조화된 메모리 사용이 가능하다. 전이 도중에, 입력된 텍스트를 하나의 연속적인 토큰으로 바꿔주는 traversal-style 접근법을 통해 특정 과제에의 입력값을 얻는다. 이러한 적응방법은 사전학습된 모델의 구조를 바꾸는 것을 최소화한다. 우리의 실험에서 증명했다시피, 이러한 적용은 효율적으로 미세조정하여 사전학습된 모델의 구조가 최소한으로만 변하도록 해준다.  
 
 자연어추론, QA, 문장유사도, 문서 분류라는 4가지 과제에 대해서 우리의 접근법을 평가해보았다. 우리의 특정 과제와 무관한 범용모델(task-agnostic)은 개별적인 과제에 특화된 모델들보다  성능이 앞섰고, 조사된 12개의 과제 중에서 9개에서 SOTA를 달성했다. 예를 들어, 상식추론(commonsense reasoning)에서 8.9%, QA에서 5.7%, TE에서 1.5%, GLUE에서는 5.5%의 성능향상을 보였다. 우리는 또한 4가지 다른 환경에서. Zero-shot 방식의 사전학습 모델을 분석했고, 이런 모델들이 세부과제를 위한 유용한 언어적 지식을 획득한다는 사실을 증명했다.   

<br/>


## 2. Background

* NLP 준지도학습(Semi-supervised learning for NLP)
 
 우리의 연구는 자연어에서의 준지도학습에 해당한다. 이 방식은 sequence labelling이나 문서분류에의 응용에 많이 사용되고 있다. 초기의 방식은 unlabeled data에서 단어나 구절 수준의 통계를 분석하여, 지도 학습의 feature값으로 제공하는 방식이었다. 지난 몇 년간, 연구자들은 unlabeled data에서 얻은 word embedding이 다양한 과제에서 성능을 향상시킨다는 점을 증명해왔다. 그러나 이러한 방식은 단어 수준의 표현을 학습할 뿐이며, 우리는 문장 수준의 임베딩을 목표로 한다.  
 
 최근에는 미분류데이터에서 단어 수준을 넘어서는 정보를 학습하는 방식을 연구하고 있다. 미분류데이터를 통해 학습된 절 단위 또는 문장 단위 임베딩은 다양한 과제에서 텍스트 데이터를 적절한 벡터 표현으로 바꿀 수 있다.  
   
 <br/>  
 
* 비지도 사전학습(Unsupervised pre-training)  

  목표가 지도학습에서 좋은 초기값을 찾는 것일 때, 비지도학습 사전학습은 준지도학습의 특별한 경우가 된다. 초기에는 이미지분류나 회귀과제에 주로 사용되었다. 후속 연구는 사전학습이 정규화와 같이 동작하여, Deep Neural Network에서 더 나은 일반화가 가능하도록 만들어준다고 보여줬다. 최근의 연구에서는 사전학습이 이미지 분류, 음성인식, 다의어 명확화, 기계번역 등 다양한 과제에 사용되고 있다.  
  
  우리의 연구(GPT)와 비슷한 연구로는 언어모델 목적함수를 이용하여 신경망 사전학습을 하고, 지도 하에 목표 과제에 맞게 미세조정하는 것이 있다. Dai et al. / Howard and Ruder의 연구는 텍스트 분류를 향상시키기 위해 이러한 방식을 이용했다. 하지만, 사전학습이 언어적 정보를 포착했음에도 불구하고, LSTM의 이용은 예측 능력이 짧은 거리에서만 유효하도록 한정지었다. 대조적으로, transformer를 선택한 우리는 언어 구조에서 더 넓은 범위를 포착할 수 있도록 설계하였다. 더 나아가서, 우리는 또한 우리의 모델이 자연어 추론, 문맥 감지 및 내용 완성 등의 다양한 과제에서도 효율적이라는 것을 증명했다. 다른 접근법들은 목표 과제에 맞는 지도학습을 하는 도중에 사전학습된 언어 표현 정보를 부가 정보로 사용하였다. 이는 목표 과제에 맞도록 많은 양의 새로운 파라미터를 추가해야 하지만, 우리는 전이를 하면서 모델 구조의 변경을 최소화하였다.  
  
  <br/>  
 
* 보조적인 학습 목적함수(Auxiliary training objectives)  

 보조적인 비지도학습 목적함수를 추가하는 것은 준지도학습의 대안이다. Collobert and Weston의 초기 연구에서 POS tagging, Chunking(명사구 묶기), NER, 언어 모델링과 같은 다양한 보조 NLP 정보를 이용하여 문맥적 역할 라벨링(semantic role labeling)의 성능을 향상시켰다. 더 최근에 Rei의 연구는 보조 언어모델링 목적함수를 목표 과제 함수에 추가해서 sequence labeling task에서 성능향상을 얻었다. GPT도 보조목적함수를 사용하지만, 비지도 사전학습이 이미 목표 과제에 관련된 여러 언어적 정보를 이미 학습했다.

<br/>


## 3. Framework

 학습은 두 단계에 걸쳐 진행된다. 첫번째 단계는 대용량 텍스트데이터로 대용량의 언어모델을 만드는 것이다. 두번째 단계는 이 모델은 라벨링한 데이터가 있는 개별적인 과제에 맞추기 위해 세부조정을 하는 것이다.  

 <br/>

### 3.1. Unsupervised pre-training  

 U=u1,…,un 이라는 비지도 말뭉치가 주어졌을 때, 우리는 표준언어모델링 함수를 사용하여 확률값을 최대화한다. 
 
 <img src="https://user-images.githubusercontent.com/11614046/105433384-d1d54280-5c9c-11eb-814d-5322cb41d4ae.png" width="50%">

 여기서 k는 문맥 범위(context window) 사이즈이고, 조건부확률 P는 parameter가 Θ인 신경망 모델을 이용하여 구한다. 이 파라미터들은 SGD(경사하강법)에 의해 학습된다.  

 우리의 연구에서, 우리는 Transformer를 변형하여, decoder 레이어를 여러개 사용하였다. 이 모델은 입력된 token들에게 multi-headed self-attention을 적용하고, position-wise feedforward layer를 적용하여 출력값을 얻는다. 식으로 표현하면 다음과 같다.  

 <img src="https://user-images.githubusercontent.com/11614046/105433965-fd0c6180-5c9d-11eb-9e79-726f5635163f.png" width="50%">

 여기서 문맥토큰은 U=u-k,…,u-1이며, n은 레이어의 개수, We는 token embedding matrix, Wp는 position embedding matrix이다.
 
 _(논문에서 ∀l이 ∀i로 오타가 나 있다. 위의 사진에서는 수정했다.)_  

 <br/>

### 3.2. Supervised fine-tuning

 3.1과 같이 목적함수를 통해 모델 훈련을 시킨 뒤에, 모델의 파라미터 값들을 특정 과제에 맞도록 미세조정해준다. 분류된 데이터셋 C가 있고, C의 각 요소들은 x1~xm의 토큰들과 정답 y로 구성되어 있다. 이 입력값들은 사전학습된 모델로 입력되어서 transformer block의 활성화값 hlm을 얻고, 이 결과는 parameter Wy와 함께 선형 결과값 레이어로 전달된다. P값은 다음 목적함수를 통해 확률값을 최대화한다. 

 <img src="https://user-images.githubusercontent.com/11614046/105434680-62148700-5c9f-11eb-8c3c-3dea806380f3.png" width="50%">  

 <br/>  

 우리는 언어 모델을 보조적인 목적함수로 미세조정에 사용하면 좋은 점을 2가지 발견했다.  
 
 - 지도학습 모델의 일반화를 향상시킨다.  
 - 수렴을 가속화한다.  
 
 이렇게 보조 목적함수가 성능을 향상시킨다는 사실들은 이전의 다른 연구에서도 밝혀진 사실이다. 특히, 우리는 아래의 목적함수를 weight λ를 사용하여 최적화한다.
 
  <img src="https://user-images.githubusercontent.com/11614046/105435194-3cd44880-5ca0-11eb-8a9a-295692098b67.png" width="50%">  

 결과적으로 보면, 미세조정을 하면서 추가로 필요로하는 파라미터는 Wy와 구분자 token을 위한 embedding뿐이다.  
 
 <br/>

### 3.3 Task-specific input transformations

텍스트 분류와 같은 몇몇 과제에서, 우리는 위에서 언급한 방법으로 모델을 쉽게 미세조정할 수 있었다. QA나 TE와 같은 몇몇 과제는 한 쌍의 문장이나 문서, 질문, 대답으로 이뤄진 3개의 텍스트가 하나의 입력값이 되어서, 구조화된 값이 입력된다. 우리의 사전학습 모델이 연속적인 텍스트의 입력으로 학습되었기 때문에, 이러한 과제에 맞도록 모델을 조금 변경해줄 필요가 있다. 이전의 연구들은 전이된 embedding 위에 과제에 맞는 아키텍쳐들을 얹어서 학습하는 방법이었다. 이러한 접근법은 많은 과제에 특화된 변형이 필요했고, 아키텍쳐 부분에서는 전이학습이 사용되지 못하였다. 하지만, 우리는 traversal-style접근법을 통해 구조화된 입력값을 우리의 사전학습된 모델이 처리할 수 있는 형태의 입력값으로 변경하였다. 이러한 입력값 변형은 과제마다 아키텍쳐를 엄청나게 바꿀 필요가 없도록 만들었다. 아래의 그림 Figure 1에서 이러한 입력값 변형을 시각적으로 묘사했다. 모든 변형은 시작과 끝 토큰인 (s), (e) 를 랜덤하게 추가한다.  
 
<img src="https://user-images.githubusercontent.com/11614046/105435376-9a689500-5ca0-11eb-96cc-ffb9b5c05900.png" width="70%">

<br/>

- **Textual entailment**

 함의 과제에서는 전제 p와 가정 h를 구분자 $로 연결하였다.   

- **Similarity**

 유사도 과제에서, 두 개의 텍스트 사이에 순서가 없다. 따라서, 텍스트 두 개를 다른 순서로 이어붙여 총 2개를 입력으로 쓴다(중간에 구분자를 넣는다). 이는 각각 정방향으로 representation으로 변화되어,  Transformer에 입력으로 들어간다.  

- **QA and Commonsense reasoning**

 이 과제를 위해서, 문맥 문서 z, 질문 q, 가능한 답변 {ak}가 주어진다. 문맥문서(documnet context)와 질문을 결합하고, 가능한 답을 더한다. 그 사이에 구분자 $를 추가하여, [z; q; $; a_k]와 같은 형태로 입력값을 생성한다. 각 입력 값들은 독립적으로 처리되며, 입력의 개수는 답변의 개수만큼 생성된다.  
 
 <br/>

## 4. Experiments

### 4.1 Setup

**Unsupervised pre-training**  

 언어모델을 학습하기 위해서 우리는 BookCorpus데이터를 사용한다. 여기에는 7000권이 넘는 어드벤쳐, 판타지, 로맨스와 같은 다양한 장르의 책들이 포함되어 있다. 이 데이터들은 긴 길이의 텍스트들을 포함하기 때문에 generative model이 긴 범위의 정보를 학습할 수 있다. 대안이 되는 데이터셋으로는 ELMO에서 사용한 1B Word Benchmark가 있는데, 이는 문장들이 섞여있어서 장거리 의존성이 파괴되어 있다. 우리의 언어모델은 이 데이터셋에서 낮은 복잡성인 18.4를 얻었다.  
 
 <br/>

 **Model Specification**  

우리의 모델은 Transformer의 초기 세팅을 거의 따른다. Transformer의 디코더만을 사용해서 masked self-attention head(768차원, 12개 attention head)로 12layer를 구성했다. (Transformer에서는 decoder가 6번). 구체적인 내부 스펙은 다음과 같다. 

<img src="https://user-images.githubusercontent.com/11614046/105787008-6f908080-5fc1-11eb-8d85-890408a17ba8.png" width="80%">  

- 추가적으로, 모든 레이어에서 Layernorm을 사용하였기 때문에 weight의 초기값은 N(0, 0.02)로 설정한다.  
- Bytepair Encoding(BPE)를 사용하여, 토큰화한다.
- residual, embedding, attention에 0.1의 dropout을 적용한다.
- positional embedding을 사용
- BookCorpus를 cleansing하기 위해 ftfy라이브러리 사용, 공백과 문장부호를 정규화, Spacy tokenizer를 사용.

<br/>

 **Fine-tuning details**  

비지도 사전학습에서 사용한 hyperparameter들을 그대로 사용했다. Classifier에 dropout 0.1을 추가하였다. 대부분의 과제에서, learning rate를 6.25e-5, batch-size 32를 사용했다. 세부조정은 대부분 빨리 끝났으며, 3 epoch면 대부분 충분했다. Linear learning rate decay는 warmup을 포함해 0.2%였고, λ는 0.5였다.  

<br/>

### 4.2 Supervised fine-tuning

 자연어 추론, QA, 문장유사도, 문장분류와 같은 다양한 지도학습 과제에서 실험했다. 이 중 일부는 GLUE benchmark에 포함되어 있다. 결과는 아래와 같다.
	
<br/>

 **Natural Language Inference**  
 
 <img src="https://user-images.githubusercontent.com/11614046/105789331-96e94c80-5fc5-11eb-92ee-2d4e3736a920.png" width="70%">  

 Image caption(SNLI), 문서화된 음성, 대중소설, 정부 보고서(MNLI), 위키피디아 기사(QNLI), 과학시험(SciTail), 뉴스기사(RTE) 등의 다양한 데이터로 실험하였다. 각 0.6~5.8% 정도 성능이 향상되었다.  

 <br/>

 **Question answering and Commonsense reasoning**  
 
 <img src="https://user-images.githubusercontent.com/11614046/105789544-fd6e6a80-5fc5-11eb-908e-37e0b26b1086.png" width="70%">  

 중고등학교 시험에서 나온 영어지문과 관련 질문으로 구성된 RACE dataset으로 진행하였다. 또 Story Cloze에 대해서도 진행했는데 이는 무려 8.9%까지 높은 성능을 내며 결과를 종합했을 때 GPT가 넓은 범위에 걸친 문맥 정보도 잘 포착해냄을 알 수 있다.  

 <br/>

 **Semantic Similarity**  

 <img src="https://user-images.githubusercontent.com/11614046/105789735-4e7e5e80-5fc6-11eb-8917-6d096d31cd53.png" width="70%">  

 QQP에 대해서는 BiLSTM + ELMo + Attention을 사용한 모델보다도 특히 성능이 향상되었다. STSB에서도 SOTA를 달성했다.
 
 <br/>

 **Semantic Similarity**  

 <결과값은 위의 사진에서 확인>

 두 개의 다른 텍스트 분류 과제에 대해서도 평가를 진행했다. CoLA(The Corpus of Linguistic Acceptability)는 어떤 문장이 문법적으로 옳은지를 전문가가 평가한 답변과, 학습된 모델에 대한 언어적 편향에 대한 테스트를 포함한다. SST-2(The Stanford Sentiment Treebank)는 표준 이진 분류 문제이다. CoLA에 대해서는 35.0점에서 45.4점으로 상승하여 SOTA를 얻었으며, SST-2에서는 91.3점으로 SOTA와 비교해봤을 때 경쟁력있는 점수를 얻었다. GLUE에서도 72.9점으로 전보다 상승했다.  
  
 <br/>
 
 결과적으로 보면, 12개의 데이터셋 중에서 9개에서 SOTA를 달성하였다. 우리의 결과는 작은 데이터셋인 STS-B(train:5700개)부터 큰 데이터셋인 SNLI(train:55만개)까지 다양한 사이즈의 데이터에서 모두 훌륭하게 작동한다는 것을 보여준다.

## 5. Analysis  

 **Impact of number of layers transferred**  

 아래 Figure 2의 왼쪽은 layer의 수를 다르게 하면서 RACE와 MultiNLI에 대해 실험을 진행한 것이다. transferring embedding이 성능 향상을 가져오며, 레이어의 개수가 늘어날수록 하나의 transformer layer 당 9%의 성능향상이 이뤄졌다. 이는 사전학습된 모델의 각각의 layer가 문제를 푸는 데 있어 유용한 기능을 한다는 것을 의미한다.  
 
 <img src="https://user-images.githubusercontent.com/11614046/105929657-69ff6d00-608b-11eb-9fdf-c6fa66fabc5a.png" width="70%">  
 <br/>
 <br/>

 **Zero-shot Behaviors**  

 왜 Transformer를 활용한 사전학습 언어모델은 효율적인 것일까? 가설을 세워보면, generative model이 언어 모델링 능력을 향상시키기 위해 많은 taskf를 수행하는 법을 배울 수 있다는 것과 LSTM보다 transformer의 attentional memory가 전이에 더 도움을 준다는 것이 있다. 귀납적인 풀이를 위해서 우리는 generative model이 과제맞춤-finetuning없이 과제들을 수행하도록 해보았다. 위의 그림 오른쪽을 보면, 사전학습을 오래할수록 모든 과제에 있어서의 성능이 안정적이고 꾸준하게 상승한다는 것을 알 수 있다. finetuning없이 pre-training만을 실시한 LSTM은 더 높은 변동성을 보여줬으며, Transformer의 구조보다 전이에 유리하지 않다는 점을 알 수 있다.  
 
 <br/>

 **Ablation studies**  

3종류의 ablation study(특정 부위를 제거하여 해당 부위의 기능을 알아보는 연구)를 진행했다. 결과는 아래의 표와 같다.  

 1.미세조정을 할 때, 보조적인 LM 목적함수를 제거한다.
  - 보조적 목적함수는 NLI과 QQP에서 성능향상을 보여줬다. 큰 데이터셋에서는 이점이 있지만, 작은 데이터셋에서는 그렇지 않았다.

 2.Transformer를 같은 구조의 LSTM(2048unit)로 대체하였는데 5.6점의 점수 하락이 있었다. 
  - 성능이 좋은 경우는 MRPC 뿐이었다.

 
 3.Transformer를 사전학습 없이 바로 지도학습을 하도록 해보았다.
 - 사전학습의 부족은 전체적으로 14.8% 정도의 성능 저하를 가져왔다. 

 <img src="https://user-images.githubusercontent.com/11614046/105931966-82718680-608f-11eb-8d34-8b746c8b2f3a.png" width="70%">  

 <br/>


## 6. Conclusion

 생성적 사전학습과 개별적인 미세조정을 통해 만들어지는 과제에 대해 별다른 지식이 없지만, 자연어이해 능력이 뛰어난 단일 모델(framework)를 소개하였다. 넓은 범위에 걸친 다양한 텍스트 말뭉치에 대해 사전학습을 진행하였고, 이를 통해 QA, 문장유사도 평가, 함의 확인, 문서분류 등의 과제에서 성공적으로 장거리 의존성을 전이하는 능력을 학습하여 12개 중 9개의 과제에 대해 state-of-the-art를 달성하였다. 개별적인 과제에 대한 성능을 높이기 위해 비지도 사전학습을 이용하는 것은 기계학습 연구의 중요한 목표가 되었다. 우리의 연구는 상당한 성능향상이 정말로 가능하며, 어떤 모델(Transformers)과 dataset(장거리 의존성을 포함하는 텍스트)이 이런 접근법에 가장 좋은지 알려준다. 이 연구가 자연어이해와 다른 분야에서의 비지도학습이 새로운 연구에 도움이 되기를 희망하며, 나아기 비지도학습이 언제, 어떻게 작동하는지에 대한 이해를 증진시키기를 바란다.
 
 <br/>
 
***
**개인적인 감상**

이 논문은 BERT와 비슷한 시기에 나왔지만, 덜 유명한 논문이다. 왜냐하면 BERT가 다양한 task에 대해 더 뛰어난 결과를 보여주기 때문이다. 사실, BERT와 GPT모델은 서로 pre-training 기법이 약간만 다른 모델이다. Bert는 encoder를 사용해 문장 전체를 보고 학습하지만, GPT는 decoder를 사용해 정방향으로만 학습을 진행한다. 하지만 GPT모델은 language model 기반이기에 BERT보다 언어 생성에 유리하다는 장점이 있으며, 이는 최근에 나온 GPT3의 위력을 통해 확인 가능하다. 그래도 기존의 LSTM에서 벗어나 Transformer를 기반으로 pre-training + fine-tuning이라는 방식을 제시했다는 점에서 의미가 있는 논문이라고 생각한다.

 <img src="https://user-images.githubusercontent.com/11614046/105942620-ebaec500-60a2-11eb-89f8-564d13d91886.png" width="50%">  


## Reference

https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/21/OpenAI-GPT-1-Improving-Language-Understanding-by-Generative-Pre-Training/  

https://www.quantumdl.com/entry/12주차1-Improving-Language-Understanding-by-Generative-Pre-Training  

https://www.youtube.com/watch?v=FeEmmylAF0o  
