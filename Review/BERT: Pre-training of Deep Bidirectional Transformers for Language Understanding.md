# 「BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding 」 Review  

![image](https://user-images.githubusercontent.com/11614046/107447361-bebedf80-6b83-11eb-9fba-efce99dabe71.png) 

https://arxiv.org/abs/1810.04805

이 논문 역시 구글에서 발표한 논문이다.

논문 제목에 나와있는 Transformer는 attention is all you need에서 공개된 모델의 이름이다. 즉 attention is all you need의 논문을 참조하고 발전시켜, Bidrectional(양방향)으로 language를 학습한 논문이다.
 
또한 당시 language 모델에서 가장 좋은 성능을 보였던 ELMo, OpenAI GPT를 겨냥한 논문으로 보인다.

<br/>


## 0. Abstract

 이 논문에서 Bidirectional Encoder Representation from Transformers(트랜스포머를 통한 양방향 인코더 표현)을 줄인 BERT를 새로운 언어 표현모델로 제시한다. 최근의 다른 언어 표현모델과는 다르게, BERT는 미분류 텍스트를 모든 레이어에서 양방향으로 학습해서 깊은(deep) 양방향 표현을 사전학습하도록 설계되었다. 결과적으로 사전학습된 BERT는 위에 하나의 레이어를 추가하고 미세학습을 한다면 SOTA모델을 설계할 수 있으며, QA 언어추론 등 다양한 과제 맞춤형 구조 변경을 할 수 있게 된다. 
 
  BERT는 개념적으로 간단하고 경험적으로 강력하다. BERT는 11개의 과제에서 SOTA를 달성했으며, 세부 과제들의 점수는 GLUE score 80.5%(7.7%상승) MultiNLI 86.7%(4.6% 상승) SQuAD v1.1 93.2(1.5점 상승) SQuAD v2.0 83.1(5.1점 상승)이다. 
  
<br/>


## 1. Introduction

 LM(Language Model)을 사전학습하는 것은 많은 자연어 과제에서 효과적이라는 사실이 밝혀졌다. 이는 문장 단위의 과제로 문장들을 전체적으로 분석해서 관계를 예측하는 자연어추론과 요약뿐만 아니라 토큰 단위의 과제로 모델들이 토큰 단위의 정제된 결과값을 요구하는 NER, QA에서 모두 효율적이다. 
 
 사전학습된 language representation을 하위의 세부 과제에 적용하는데에는 2가지 전략(Feature-based & Fine-tuning)이 있다. ELMO와 같은 feature-based방법은 사전학습된 값들을 추가적인 feature로 사용하여 task-specific architecture에 추가하는 방법이다. GPT와 같은 fine-tuning방법은 task-specific한 파라미터들을 최대한 줄여서, pertained 파라미터를 fine-tuning하여 사용하는 방법이다. 두가지 접근방법은 사전학습 도중에 동일한 목적함수를 사용하며, 단방향(unidirectional) 언어모델(LM)을 사용하여 언어표현(language representation)을 학습한다. (구글에서는 이 점을 단점 및 문제점이라고 인식한 것 같다.) 
 
 우리는 현재의 테크닉이 사전학습의 성능 특히 미세학습 접근방식의 성능을 제한한다고 생각한다. 주요 제한점은 LM이 단방향이라는 점이며, 이 점이 사전학습 도중에 아키텍쳐의 학습을 제한한다고 생각한다. 예를 들어, GPT의 설계자들은 왼쪽에서 오른쪽으로만 읽을 수 있도록 설정해서 모든 토큰들이 자신의 이전 토큰들만을 attention할 수 있다. 이러한 제한은 문장 단위의 과제에서 최적화가 제대로 되지 않으며, 토큰 단위의 과제에서도 QA와 같이 양 방향에서 문맥을 읽어내야하는 경우에는 해롭다. 
 
 이 논문에서, 우리는 BERT를 통해 fine-tuning 방식의 접근법을 개선한다. BERT는 “Masked Language Model(MLM)”이라는 목적 함수를 통해 앞에서 언급한 단방향성으로 인한 문제점을 개선한다. MLM은 입력된 토큰들의 일부분을 랜덤하게 가려서, 문맥을 통해 가려진 토큰의 id값을 추론하도록 만드는 방법이다. 왼쪽에서 오른쪽으로 읽는 모델의 사전학습과는 다르게, MLM은 양방향의 문맥을 다 읽을 수 있으며 양방향 Transformer를 통해 사전학습할 수 있도록 설계되었다. MLM에 추가적으로 next sentence prediction task를 하도록 했다. 그 결과는 다음과 같다. 
 
 - 언어적 표현에 있어서 양방향 사전학습의 중요성에 대해서 보여주도록 하겠다. 다른 일방향 모델과 달리 BERT는 masked language model을 통해 양방향 사전학습이 가능하다. 이는 왼쪽에서 오른쪽 또는 오른쪽에서 왼쪽으로 독립적으로 진행하여 결합시키는 LM과는 다르다. 
 - 사전학습된 표현방식이 과제 특화구조를 심각하게 설계할 필요가 없음을 보여주도록 하겠다. BERT는 문장 단위 또는 토큰 단위의 과제에서 간단한 미세조정만으로도 과제에 최적화된 설계를 능가하는 SOTA 성능을 처음으로 보여준 방식이다. 
 - BERT는 11개의 NLP과제에서 SOTA를 달성했다. 코드와 사전학습 모델은 http://github.com/google-research/bert 에서 볼 수 있다.
 
 <br/>


## 2. Background

 사전학습을 통해 일반화된 언어모델을 만드는 것은 오랜 역사를 가지고 있고, 이 섹션에서는 이러한 역사를 간단하게 리뷰하도록 하겠다.

### 2.1 Unsupervised Feature-based Approaches

 다양하게 사용될 수 있는 언어의 표현은 수십년간 활발하게 연구되어온 분야이다. 사전학습된 단어 임베딩은 최신 NLP 시스템에서 핵심적인 역할을 하고 있다. 단어 임베딩 벡터들을 사전학습하기 위해 왼쪽에서 오른쪽으로 읽는 언어모델링 목적함수가 사용되었을 뿐만 아니라 왼쪽에서 오른쪽 문맥을 읽어가면서 옳은 것들 중에서 틀린 것을 찾는 목적함수 역시 존재한다.

 이러한 접근법들은 문장 임베딩아나 단락 임베딩과 같이 뭉뚱그려진 접근법으로 일반화되어 왔다. 문장 표현을 학습하기 위해, 이전의 연구들은 다음 문장 후보군들을 ranking하거나, 왼쪽에서 오른쪽으로 읽어가면서 다음 문장을 생성하거나, auto-encoder를 활용하여 denoising하는 방식의 목적함수를 사용했다.

 ELMO와 이전의 모델들은 다양한 차원으로 단어를 임베딩했다. 이들은 문맥적 특징들을 반영하여, 양방향 LM을 만들었다. 각 토큰들의 문맥적 표현들은 각 방향에서 나온 결과들을 concat하여 만들었다. 기존에 있었던 과제 특화 설계구조에 이러한 문맥적 단어 임베딩을 input값으로 사용했더니, 몇몇 주요 NLP과제(QA, 감성분석, NER)에서 SOTA를 달성할 수 있었다.
 
 <br/>  

### 2.2 Unsupervised Fine-tuning Approaches

 Feature-based 접근방식처럼, 사전학습모델을 활용하려는 시도는 2008년부터 시작되었다.  

 더 최근에는 문맥적 토큰 표현을 생성하는 문장 또는 문서 인코더가 unlabeled text로부터 사전학습되고, 특정 하위 과제에 맞도록 미세조정되도록 만들어졌다. 이러한 접근법들의 장점은 초기에 몇 개만 학습하면 된다는 점이다. 이러한 장점때문에 OpenAI GPT는 많은 문장단위 과제에서 SOTA를 달성 할 수 있었다. 왼쪽에서 오른쪽으로 읽는 언어 모델과 auto-encoder 목적함수는 이러한 모델의 사전학습에 사용되어 왔다.  

 <br/>  

### 2.3 Transfer Learning form Supervised Data

 Natural Language Inference나 기계번역과 같이 대용량 데이터셋으로부터 지도학습을 진행한 뒤에 효과적으로 전이학습을 한 사례들이 존재한다. Vision 분야에서도 ImageNet으로 사전학습하여 미세조정한 모델이 효과적인 방법임을 보여주고 있다. 

 <br/>


## 3. BERT

 이번 섹션에서는 BERT를 소개하고, 세부적인 내용을 설명하겠다. 우리의 설계(Framework)는 사전학습과 미세조정이라는 2단계로 이뤄진다. 사전학습에는 unlabeled된 데이터들을 다른 사전학습 과제를 통해 모델을 학습시킨다. 미세조정에서는 사전학습을 통해 얻어진 파라미터로 시작해서, 모든 파라미터들을 labeled된 데이터로 특정 과제에 맞도록 미세조정해준다. Figure 1에서는 QA모델이 어떻게 만들어지는지 표현되고 있다. 
 
 BERT의 독특한 특징은 다양한 과제들에서 공통적인 구조를 사용한다는 점이다. 사전학습 모델과 하위 과제 모델에는 최소한의 차이만이 존재한다.

 <img src="https://user-images.githubusercontent.com/11614046/107898364-e2689800-6f7e-11eb-872a-65369bc4e55c.png" width="80%">

 <br/>  
 <br/>  
 
 **Model Architecture**
 
 BERT의 모델 구조는 multi-layer bidirectional Transformer이며, tensor2tensor 라이브러리에 구현된 것을 이용하였다. Transformers의 사용이 흔해졌고, 우리의 적용도 기존의 모델과 거의 동일하기 때문에 모델 구조의 배경적인 설명은 Vaswani의 “The Annotated Transformer”를 참고하면 될 것 같다. 
 
 이 논문에서, 레이어의 개수(Transformer 개수)는 L, 히든 사이즈는 H, self-attention head의 개수는 A라고 표현하겠다. Bert base에서는 L=12, H=768, A=12, 총 파라미터의 개수는 110M였으며, Bert Large에서는 L=24, H=1024, A=16, 총 파라미터의 개수는 340M이었다.

 Bert Base는 OpenAI의 GPT와 성능을 비교하기 위해서 같은 모델사이즈로 설정되었다. 그러나 Bert의 Transformer는 양방향 self-attention을 활용한 반면에 GPT의 Transformer는 자신 이전에 나온 토큰들(왼쪽방향)만을 attention할 수 있도록 제한했다는 차이점이 존재한다.  

 <br/>

 **Input/Output Representaions**
 
 BERT가 다양한 하위 과제를 다룰 수 있도록 하기 위해서, 우리의 input representation은 single sentence와 pair of sentence를 하나의 token sequence에서 명확하게 표현할 수 있도록 하였다. 이를 통해 문장은 실제 언어학적인 sentence가 아닌 임의적으로 연속된 텍스트로 표현이 된다. Sequence는 BERT에 들어가는 input token sequence를 의미하며, single 또는 two sentences가 들어갈 수 있다. 
 
 우리는 3만개의 토큰을 지원하는 WordPiece embedding을 이용했다. 모든 sequence의 첫 토큰은 CLS를 사용한다. 이 토큰에 대응하는 final hidden state 값은 Classification task에서 사용할 수 있다. Sentence pair는 하나의 sequence로 묶인다. 각 문장들은 2가지 방법으로 구분한다. 첫번째로, 문장들은 SEP 토큰으로 분리한다. 둘째로, 임베딩을 할 때 해당 토큰이 문장 A인지 B인지 알려주는 값들을 넣어준다. Figure1에서 보이는 것과 같이 입력 임베딩 값을 E로 표현하고, CLS토큰의 final hidden vector값은 C, i번째 입력 임베딩 값의 final hidden vector는 Ti 로 표현한다.  
 
 주어진 토큰에 대해서, input representation은 해당되는 토큰값 + 문장번호 + 위치로 이뤄진다. 이런 방식은 Figure 2에서 볼 수 있다.  
 
 
### 3.1. Pre-training BERT

 다른 모델들과 달리 BERT를 사전학습하기 위해서 전통적인 방식처럼 왼쪽에서 오른쪽으로 보거나, 오른쪽에서 왼쪽으로 보지 않았다. 대신에, 우리는 이번 섹션에서 설명할 2가지 비지도학습을 이용해 BERT를 사전학습하였다.  
 
 <br/>  

 **Task #1: Masked LM**

 직관적으로도 left-to-right 또는 right-to-left로 읽어서 합치는 모델보다는 deep bidirectional 모델이 더 강력하다고 생각하는 것은 합리적이다. 불행히도, 기존의 LM에서는 bidirectional하면 간접적으로 타겟단어들을 보게되기 때문에 left-to-right 또는 right-to-left로만 학습할 수 밖에 없었으며, 모델들이 다음의 단어를 예상하는 방식으로 학습되었다.  
 
 Deep bidirectional representation을 학습하기 위해, 우리는 단순히 입력 토큰의 일부를 마스킹했고, 마스킹된 토큰을 예측하는 방식으로 학습시켰다. 이러한 과정을 masked LM (MLM)이라고 부른다. 이 경우에, 마스킹 토큰에 대응하는 final hidden vector를 vocab에 대한 output softmax에 전달된다. 각 sequence에서 15% 정도의 확률로 WordPiece token을 랜덤하게 마스킹했고, 우리는 denoising auto-encoders와는 다르게 전체 input을 reconstructing하기보다는 masked words만 predict했다.  
 
 MLM 덕분에 bidirectional한 사전학습 모델을 만들 수 있게 되었지만, 문제점은 미세조정에는 MASK가 발생하지 않기 때문에 사전학습과 미세조정 간의 불일치가 만들어진다는 점이다. 이러한 문제를 완화하기 위해서, 우리는 마스킹된 단어를 항상 발생시키지 않았다. 학습 데이터 생성자는 15%의 확률로 MASK토큰을 만들어낸다. 만약에 i번째 토큰이 마스킹되기로 결정되면, 80%만 MASK로 치환하고, 10%는 랜덤으로 치환하고, 10%는 바꾸지 않고 놔둔다. 그리고, Ti는 cross entropy loss를 활용하여 원래의 토큰을 예측한다. 이 과정은 부록 C2에서 변형해서 비교해놨다.  
 
 <br/>

 **Task #2: Next Sentence Prediction (NSP)**
 
 QA나 NLI와 같은 많은 중요한 세부과제들은 LM에서는 직접적으로 포착되지 않는 두 문장간의 관계를 이해하는 것이 필요하다. 문장간의 관계를 이해하도록 모델을 학습시키기 위해서, 우리는 단일 언어 말뭉치로 binarized된 NSP를 사전학습시킨다. A, B문장이 있을 때, 50%의 확률로 B는 A와 바로 이어지는 문장(IsNext)이고 50%의 확률로 B는 말뭉치 내에서 랜덤으로 골라진 A와 관계없는 문장(NotNext)이다. Figure1에서 보이는 것처럼 C는 NSP를 하는데 이용된다. 단순하지만, Section 5.1에서 증명하는 것처럼 QA와 NLI에서 효과적임을 보여주었다. NSP는 Jenite et al. (2017)과 Logeswaran and Lee (2018)에서 사용한 representation-learning objectives와 유사하다. 그러나, 문장임베딩을 세부 과제에 이용한 이전의 논문들과는 다르게 BERT는 모든 파라미터들을 그대로 유지하면서 세부 과제를 학습시작한다.  
 
 <br/>
 
 **Pre-training data**

 사전훈련 과정은 LM에서 수행하는 것   과 거의 비슷하다. 사전학습 말뭉치로 BooksCorpus(800M words)와 영어 위키피디아(2500M words)를 사용했다. 위키피디아와 같은 경우에는 본문만 사용하고, list table header와 같은 정보들은 제외했다. 문장수준의 말뭉치를 사용하지 않고, 문서단위의 말뭉치를 사용해서 long contiguous sequences를 학습시키고 싶었다.  
 
 <br/>

### 3.2 Fine-tuning BERT

 Transformer의 self-attention 덕분에 BERT에 1~2문장을 하위 과제에 맞는 형식으로 입력해주면 미세조정을 직관적으로 할 수 있다. 일반적으로 텍스트 pair를 적용할 때,텍스트 pair를 독립적으로 한 문장씩 인코딩하여 bidirectional cross attention을 해준다. 하지만 BERT는 문장쌍을 한번에 인코딩하여, 두 문장 사이의 bidirectional cross attention을 포함해서 self-attention도 구해줄 수 있다.  
 
 각 과제에서 우리는 단순히 과제에 맞는 입력값과 결과값을 BERT에 넣어서, 모든 파라미터들을 미세조정시켰다. 입력부분에서, 사전학습에서 문장 A와 B는 (1) 요약에서의 문장쌍 (2) 함의에서의 가설-전제쌍 (3) QA에서의 질문-문단쌍 (4) 텍스트분류 및 시퀀스 태깅에서의 텍스트-0쌍과 유사하다. 결과값에서, 토큰 값들은 시퀀스 태깅이나 QA와 같은 토큰 단위의 과제에서 output layer에 입력되며, CLS값은 entailment나 sentiment analysis와 같은 분류 과제에서 output layer에 입력된다.  
  
 사전학습과 비교해서, 미세조정은 상대적으로 비용이 싸다. 이 논문에서의 모든 결과들은 동일한 사전학습 모델을 가지고 single Cloud TPU에서 1시간, GPU로 몇시간이면 재현할 수 있다. 우리는 세부 과제들의 상세 내역들을 section 4에서 언급할 것이다. 더 자세한 내용은 부록 A.5를 참고하면 된다.  
         
<br/>


## 4. Experiments

 이 섹션에서, 우리는 11개의 NLP과제에 대해서 BERT를 세부조정한 결과를 보여주겠다.
 
### 4.1 GLUE

 General Language Understanding Evaluation (GLUE)는 다양한 자연어 이해 과제들을 모아놓은 것이다. GLUE 데이터셋에 대한 자세한 설명은 부록 B.1에 있다.  
 
 GLUE를 미세조정하기 위해서, 우리는 입력값을 섹션3에서 언급된바와 같이 변형하였으며, CLS토큰에 대응하는 final hidden vector C ∈ 𝑹^H 값을 이용했다. 미세조정 중에 새롭게 추가된 파라미터는 classification layer weight인 W∈ 𝑹^(K*H)이고, K는 라벨의 개수를 의미한다. 그리고 C와 W를 이용하여 classification loss를 계산했다. 예를 들면 log(softmax(C * W^T))  

 <img src="https://user-images.githubusercontent.com/11614046/108445409-b3646600-729f-11eb-97cc-5e4e0dc04ece.png" width="60%">  
 
 결과는 위와 같다. 주목할만한 것은 GPT와 BERT base가 구조는 비슷하지만, attention masking을 사용했다는 점만 다른데 성능은 BERT가 4.6% 앞선다. BERT Large는 BERT Base보다 모든 면에서 성능적으로 우월했다. 모델 사이즈에 따른 연구는 Section 5.2에서 추가적으로 알아보도록 하겠다.  
 
 <br/>  
 
### 4.2 SQuAD v1.1

 Standford Question Answering Dataset(SQuAD v1.1)은 100k의 질의응답쌍의 데이터이다. 질문이 주어지 답을 포함하는 위키피디아의 문단이 있어서, 며과제는 문단으로부터 정답 텍스트를 찾아내면 된다.  
 
 Figure1에서 보인바와 같이, QA과제에서 우리는 입력 질문과 문단을 하나의 sequence로 표현했으며, 질문은 embedding A 문단은 embedding B로 표현했다. 그리고 미세조정을 할 때, 시작 vector를 S ∈ 𝑹^H, 끝 vector를 E ∈ 𝑹^H로 추가했다. 그리고 타겟 단어 i의 시작 토큰일 확률은 Ti와 S를 dot-product하여 softmax하여 계산한다. 유사한 식으로 끝 토큰일 확률을 구한다. 위치가 i부터 j까지인 후보 span의 점수는  S*Ti + E*Tj를 하고, j가 i보다 크면서 점수가 최대인 것을 예측값으로 선정한다.  훈련 목적함수로는 log-likelihood를 사용한다. 
 
 <img src="https://user-images.githubusercontent.com/11614046/108447119-aa28c880-72a2-11eb-8977-e43199697104.png" width="60%"> 
 
 결과는 위와 같다. 역시나 BERT Large가 wide margin을 통해 가장 성능이 좋았음을 알 수 있다.

 <br/>  
 
### 4.3 SQuAD v2.0

 SQuAD 2.0은 SQuAD 1.1을 확장하여 답이 없는 경우도 추가했다. 
 
 기본적인 구조는 SQuAD 1.1과 동일하게 했지만, 정답이 없는 경우에는 정답의 시작과 끝이 CLS토큰을 가르키도록 설정했다. 
 
 <img src="https://user-images.githubusercontent.com/11614046/108447678-a2b5ef00-72a3-11eb-9069-03752f859c11.png" width="60%"> 

 이전의 SOTA보다 성능이 F1 score 5.1이 개선된 것을 알 수 있다.
 
 <br/> 
 
### 4.4 SWAG

 Situations With Adversarial Generations(SWAG) 데이터셋은 11.3만개의 sentence-pair로 이뤄져있으며, grounded common sense inference를 측정한다. 하나의 문장이 주어지고, 4개의 후보문장 중에 가장 잘 이어지는 1개의 문장을 찾는 과제이다. 예를 들면 다음과 같다. A girl is going across a set of monkey bars. She (i) jumps up across the monkey bars. (ii) struggles onto the bars to grab her head. (iii) gets to the end and stands on a wooden plank. (iv) jumps up and does a back flip.  
 
 SWAT데이터셋으로 미세조정을 할 때, 4개의 입력 sequence를 만들었는데 각각은 주어진 문장(문장A)과 가능한 후보문장(문장B)를 합쳐서 만들었다. 이 과제에만 사용되는 파라미터에는 CLS토큰에 대응하는 벡터값 C를 softmax하여 정규화한 값이 있다.  

 <img src="https://user-images.githubusercontent.com/11614046/108644623-20236e80-74f3-11eb-91a3-e7a2541859da.png" width="60%"> 
 
 우리는 이 모델을 2e-5의 learning rate, batch-size 16, epoch 3로 미세조정하였다. 결과는 Table 4와 같다. BERT Large는 기존의 baseline인 ESIM+ELMOfmf 27.1%, GPT를 8.3% outperform한다.  
 
 <br/>  
 
## 5. Ablation Studies

 이 섹션에서는 중요한 부분이라고 했던 부분들을 하나씩 제거하면서(=Ablation study), BERT 각 요소들의 상대적인 중요도를 이해해보도록 하겠다. 추가적인 내용은 부록 C에서 확인할 수 있다.  
 
 (개인적으로 이 논문에서 가장 중요한 부분이라고 생각한다.)

### 5.1 Effect of Pre-training Tasks

 BERT base에서의 hyperparameter, 동일한 pre-training data, 미세조정 과정을 완전히 같게하여, 학습 목적함수의 성능을 평가하고 deep bidirectionality의 중요성을 보이도록 하겠다.  
 
 <br/>
  
 **NO NSP** = MLM(Masked LM)은 하지만, NSP(next sentence prediction)은 하지 않은 모델.

 **LTR & NO NSP** = Left-context-only 모델은 MLM이 아니라 left-to-right(LTR) LM으로 학습된 모델이다. 왼쪽으로만 학습하는 것은 pre-training뿐만 아니라 fine-tuning에서도 사용했다. 왜냐하면 사전학습과 미세조정의 불일치는 세부 과제 성능에서 극심한 성능저하를 가져오기 때문이다. 추가적으로 이 모델은 NSP가 없다. 이는 OpenAI의 GPT와 거의 유사하지만, 우리의 더 큰 학습 데이터셋을 이용했으며 우리의 input representation과 우리의 fine-tuning scheme을 사용했다는 점이 다르다.  
 
 <br/> 
 
 <img src="https://user-images.githubusercontent.com/11614046/108646107-3122ae80-74f8-11eb-8ef6-1336e98229a8.png" width="80%">  
 
 첫번째로 NSP에 따른 차이점을 비교해보았다. Table 5에서 NSP를 제거하면 QNLI, MNLI, SQuAD 1.1에서 성능이 떨어지는 것을 볼 수 있다. 그 다음에, 우리는 NO NSP와 NO NSP+LTR을 비교해보았다. LTR모델은 MLM모델보다 전반적으로 모든 부분에서 성능이 떨어졌으며, MRPC와 SQuAD에서 특히 성능이 엄청 떨어지는 모습을 보였다. 

 SQuAD에서 LTR은 토큰 단위의 예측을 하며 오른쪽을 보지 못하기 때문에 성능이 떨어지는 것은 직관적으로 이해가 된다. LTR시스템을 강화하여 더 높은 신뢰도를 얻기 위하여, 우리는 randomly-initialized된 BILSTM을 맨 위에 추가해주었다. 이는 SQuAD의 성능을 상당히 향상시켜줬지만, 결과는 여전히 사전학습된 양방향 모델보다 낮은 성능이었다. BILSTM은 GLUE의 성능은 떨어뜨렸다.  

 ELMO와 같이 LTR과 RTL모델을 따로 학습하여 결합시키는 방식도 시도해보았다. 그러나 이는 (a) 양방향 모델보다 2배 더 연산량이 많으며, (b) RTL은 대답을 먼저보고 질문을 보기 때문에 QA와 같은 모델에서 직관적이지 못하며, (c) 왼쪽 오른쪽 문맥을 모든 레이어에서 사용하기 때문에 deep bidirectional model에 비해서 덜 강력하다.  
 
 <br/>

### 5.2 Effect of Model Size

 이 부분에서는, 미세조정을 하는데 있어서 모델 사이즈의 효과에 대해서 알아보도록 하겠다. 우리는 레이어의 개수, hidden unit의 개수, attention head의 개수를 다르게 하여 여러개의 BERT 모델을 만들었지만, hyperparameter와 학습 과정은 이전과 같게 유지했다.  

 <br/>

 <img src="https://user-images.githubusercontent.com/11614046/108646229-819a0c00-74f8-11eb-8567-ea8e656e5ac8.png" width="80%">  

 Table 6에서 GLUE의 성적을 볼 수 있다. 5번의 미세조정을 랜덤 시작한 결과들의 평균 값이다. 4개의 데이터셋에서 모델 사이즈가 클수록  성능이 개선되는 점이 보이며, 데이터가 3600개밖에 안되는 MRPC에서도 이러한 경향을 보였다. 이미 큰 모델을 더 크게 만들어서 더 좋은 성능을 얻을 수 있다는 점이 놀랍다. 이에 따라, BERT base는 파라미터 110M개, BERT large는 파라미터 340M개이다.  
 
 기계번역이나 LM과 같은 대규모 과제에서는 모델 사이즈를 크게할수록 성능이 계속 상승한다는 점은 오래전부터 알려져 있었다. 그러나, 모델 사이즈르 크게 할수록 적은 데이터밖에 없는 과제도 적당히 사전학습을 한다면 성능이 개선되는 점을 발견했다. 기존에 다른 연구에서는 hidden dimension의 사이즈를 200에서 600으로 늘리면 성능이 향상되었지만, 1000이 넘어가면 도움이 되지 않는다고 한 적이 있다. 이러한 결과는 과거의 연구가 feature-based approach를 했으며, 추가적인 파라미터의 일부만을 사용했기 때문인 것으로 보인다. BERT와 같은 task-specific 모델은 더 큰 pretrain모델이 있으면 하위과제의 데이터가 적은 것과 상관없이 성능이 개선되는 점을 보인다.  
    
 <br/>
 
### 5.3 Feature-based Approach with BERT
 
 지금까지 보여준 모든 BERT 결과들은 fine-tuning 접근법을 통한 결과이며, 간단한 classification 레이어를 사전학습 모델에 추가하고 사전학습에서 사용한 모든 파라미터들을 특정 과제에서 사용한 방식이다. 하지만, 사전학습 모델에서 고정된 feature값을 추출하는 feature-based 접근법은 일부 장점이 존재한다. 첫번째로, 모든 과제가 Transformer encoder 구조로 쉽게 표현할 수 있는 것은 아니기 때문에, 그런 난해한 과제들은 과제에 맞춤형으로 구조를 만들 필요가 있다. 둘째로, 학습용 데이터를 pretrain하여 비싼 연산을 한번 진행하고 나면, 많은 추가적인 실험에 대해 미리 연산된 값들을 이용하기 때문에 전체적인 연산량이 줄어든다.  
  
 이 섹션에서 우리는 BERT를 CONLL-2003 NER과제에 2가지 방식으로 적용해보도록 하겠다. BERT의 input으로는 WordPiece모델을 사용했고, 최대한의 문서의 문맥을 살리도록 했다. Output 레이어에 CRF레이어를 추가하지 않았다. 단어가 여러개의 토큰으로 잘린 경우, 첫 토큰의 태그가 단어의 태그가 되도록 설정했다.  

 Fine-tuning접근법을 제거하기 위해서, 우리는 BERT를 fine-tuning하지않고 내부에 활성화된 레이어들에서 토큰의 활성화값을 추출했다. 이러한 문맥적 임베딩은 768차원 BILSTM + classification 레이어의 input으로 사용되었다.  
 
 <img src="https://user-images.githubusercontent.com/11614046/108932272-65c76f00-768c-11eb-9c19-85948d1c133e.png" width="80%">  
 
 결과는 Table7과 같다. BERT Large는 최고의 성능을 보였다. 위쪽 4개의 encoder 값들을 concat하여 feature로 사용했을 때 최고의 성능이 나타났으며, finetuning한 결과와 0.3밖에 차이가 나지 않았다. 이는 BERT가 finetuning과 feature-based 접근법 모두에서 효과적이라는 사실을 증명한다.  
 
  <br/>
 
## 6. Conclusion  

 최근의 LM 전이학습을 통한 경험적인 성능의 향상은 풍부한 비지도 사전학습이 많은 언어적 시스템에서 중요한 부분임을 보여준다. 특히, 이러한 결과는 학습 데이터가 적은 경우에도 deep unidirectional architectures로 수혜를 볼 수 있다. 우리의 주요한 공헌은 deep bidirectional구조를 일반화된 모델로 만들어서, 다양한 NLP 과제에서 동일한 사전학습 모델을 이용할 수 있도록 한 것이다.

 <br/> 
 
## 7. Appendix for "BERT"

 이 섹션은 3파트로 이뤄져있다. 

 - Appendix A : BERT 모델에 대한 추가적인 설명
 - Appendix B : 우리의 실험에 대한 추가적인 디테일
 - Appendix C : 추가적인 Ablation Studies (Training step의 영향, 마스킹 과정의 영향)

<br/>

### Appendix A : Additional details for BERT

#### A.1 Illustarion of the Pre-training tasks

 사전학습 과제는 다음과 같다.

 **Masked LM and the Masking Procedure**  
 
 unlabel된 데이터가 My dog is hairy이며 랜덤 마스킹이 4번째 토큰인 hairy를 선택했다고 가정한다면, 우리의 마스킹 과정은 다음과 같이 진행된다.

- 80% : 해당 단어를 [MASK]토큰으로 변환한다. My dog is [MASK]
- 10% : 해당 단어를 랜덤으로 변환한다. My dog is apple
- 10% : 해당 단어를 바꾸지 않는다. 실제 관측된 단어에 대한 표현에 편향성이 생기게 하기 위해서이다. My dog is hairy

 이 절차의 장점은 Transformer encoder가 어떤 단어를 예측할지 알 수 없기 때문에, encoder가 모든 토큰들의 문맥적 표현을 강제적으로 확인해야한다는 것이다. 추가적으로 랜덤으로 바뀌는 것은 모든 토큰에서 1.5%밖에 안되기 때문에, 모델의 언어적 이해 능력을 저해하지 않는다. C2에서 이러한 절차의 효과를 알아보도록 하겠다.  

 일반적인 LM training과는 다르게 masked LM은 매 batch마다 15%의 토큰만 예측을 하게 되기 때문에 학습에 더 많은 step이 필요하다. C.1에서는 MLM이 left-to-right모델보다 천천히 학습한다는 점을 보여줄 것이다. 하지만 MLM의 성능 향상은 학습 비용 증가를 감수할만한 개선을 경험적으로 보여준다.  
  
 <br/>  
 
  **Next Sentence Prediction**  

 NSP는 다음 예제와 같이 진행된다.
 
 Input = [CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]
 
 Label = isNext
 
 Input = [CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
 
 Label = NotNext

 <br/>

#### A.2 Pre-training Procedure

 각각의 학습 입력 sequence를 만들기 위해, 우리는 말뭉치에서 2개의 텍스트 덩어리를 뽑았고 우리는 이를 1개의 문장보다는 일반적으로 훨씬 길기 때문에 ‘Sentences’라고 부른다. 첫 문장은 A embedding이며, 두번째 문장은 B embedding이다. 50%의 확률로 B는 A와 실제로 연결되는 문장이고 50%는 랜덤한 문장이며, 이를 통해 NSP 학습을 진행한다. 합친 문장의 토큰의 개수가 512개 이하가 되도록 뽑는다. LM 마스킹은 WordPiece 토큰화를 한 이후에 15%의 확률로 적용되며, 그 외에 특별한 고려사항은 없다.  

 우리는 batch size를 256 sequences로 했고 (1 sequence에 512토큰이므로, 1 batch에 256 * 512 = 128000개의 토큰을 한번에 학습한다), 총 1000000 step을 40 Epoch반복하여 3.3	 word corpus를 학습한다. 우리는 Adam optimizer, learning rate 1e-4, B1 = 0.9, B2 = 0.999, L2 weight decay 0.01, learning rate warmup 10000step, learning rate linear decay를 적용했다. 또한, GPT와 마찬가지로 relu가 아닌 gelu activation을 적용했다. Training Loss는 masked LM의 평균과 NSP의 평균을 더한 값으로 한다.  

 BERT base의 학습은 Pod configuration의 4 cloud TPU를 사용한다. BERT large는 16 cloud TPU를 사용했다. 각 훈련은 4일이 걸렸다.  
 
 긴 sequence는 불균형적으로 비쌌는데, 왜냐하면 attention이 sequence의 길이에 제곱이었기 때문이다. 우리 실험에서 속도를 빠르게 하기 위해서, 우리는 90%의 step에서 sequence length 128까지를 학습했다. 그리고, 나머지 10%의 step에서 sequence length 512까지를 학습한다.  
 
 <br/>

#### A.3 Fine-tuning Procedure

 미세조정에서, 대부분의 모델 hyperparameter는 사전학습과 동일했다. Batch size, learning rate, training epoch만 바꿨다. Dropout은 계속 0.1이었다. 최적의 hyperparameter는 과제마다 다르기 때문에, 다음과 같은 범위에서 값을 바꿔가면서 모든 과제를 수행하여 보았다.  

 - Batch Size : 16, 32
 - Learning rate(Adam) : 5e-5, 3e-5, 2e-5
 - Number of epochs : 2, 3, 4
 
 데이터셋의 크기가 커질수록 hyperparamter에 의한 성능 변화가 적었다. Fine-tuning은 일반적으로 매우 빨랐으며, 위의 파라미터들을 경우의 수대로 실행해보는 exhaustive search를 수행할 수 있었다. 최고의 성능을 이 모델의 성능으로 결정했다.  
 
 <br/>

#### A.4 Comparison of BERT, ELMO, and OPENAI GPT

 여기서 우리는 최근에 유명한 representation 방식인 ELMO, GPT, BERT를 비교해보도록 하겠다. 각 모델간의 비교는 Figure 3에서 볼 수 있다. 추가적인 차이점으로는 BERT, GPT는 fine-tuning approach이지만, ELMO는 feature-based approach라는 것이다.

 <img src="https://user-images.githubusercontent.com/11614046/109089670-f8334580-7754-11eb-9ffe-bfc6f2d4e62c.png" width="80%">  
 
 <br/>
 <br/>

 BERT와 비교하기에 가장 좋은 모델은 left-to-right Transformer LM인 GPT이다. 사실, BERT는 GPT의 많은 부분을 의도적으로 비슷하게 따라해서, 최소한의 차이만으로 비교를 할 수 있도록 설계되었다. Section 3.1에서처럼 BERT의 2가지 pretraining task(MLM, NSP)와 bidirectionality가 경험적으로 성능의 개선의 핵심인 것 같기는 하지만, BERT와 GPT가 학습될 떄 약간의 차이가 있다. 

 - GPT는 BookCorpus(800M words)로 학습되었지만, BERT는 BookCorpus(800M) + Wikipedia(2500M word)로 학습되었다.  
 
 - GPT는 문장을 분리하는 [SEP]와 classifier token인 [CLS]를 fine-tuning에서만 사용하였지만, BERT는 [SEP], [CLS], sentence A/B embedding을 pre-training에서 학습했다.  
 
 - GPT는 1M step(batch size 32000words)이지만, BERT는 1M step(batch size 128000words)이다.  
 
 - GPT는 모든 fine-tuning에서 learning rate 5e-5를 사용하였지만, BERT는 과제마다 다른 learning rate를 적용했다.  
 
 이러한 차이의 효과를 줄이기 위해서, 우리는 Section 5.1에서 Ablation experiments를 진행하여 대부분의 성능의 차이가 두가지의 pre-training task와 bidirectionality에서 온다는 점을 보였다.  
 
 <br/>

#### A.5 Illustrations of Fine-tuning on Different Tasks

 BERT를 finetuning하여 다양한 과제를 수행한 결과는 Figure 4에서 볼 수 있다. 우리의 과제 맞춤형 모델은 BERT에 output Layer 하나만을 더해주어서, parameter가 최소한으로만 변할 수 있었다. 이러한 과제들 사이에서, (a) (b)는 sequence level의 과제이며, (c) (d)는 token-level의 과제이다. Figure 4에서 E는 input embedding이고, Ti는 token i의 contextual representation이고, [CLS]는 분류를 위한 결과이며, [SEP]는 비 연속적인 토큰 sequence를 분류하기 위한 스페셜토큰이다.  

 <img src="https://user-images.githubusercontent.com/11614046/109091519-4d248b00-7758-11eb-988d-3c8a84ff974a.png" width="80%">  

 <br/>
 <br/>


### Appendix B : Detailed Experimental Setup

#### B.1 Detailed Descriptions for the GLUE Benchmark Experiments

 **MNLI** : 문장 쌍이 주어지고, 관계가 entailment, contradiction, neutral인지 고르는 classification과제.

 **QQP** : Quora에서의 두 질문이 문맥적으로 동일한지를 파악하는 binary classification.
 
 **QNLI** : 질문과 정답 쌍이 맞는지 찾는 binary classification
 
 **SST-2** : 영화 리뷰 문장이 하나 주어지고, 어떤 감정인지 찾는 classification
 
 **CoLA** : 영어 문장 하나가 주어지고, 문법적으로 옳은지 판단하는 binary classification
 
 **STS-B** : 뉴스 헤드라인과 문장이 주어지고, 얼마나 의미적으로 유사한지 1~5점을 주는 과제
 
 **MRPC** : 온라인 뉴스에서 뽑은 두 문장이 의미적으로 유사한지를 판단하는 과제

 **RTE** : MNLI와 같은 과제이지만 training data의 개수가 더 적다

 **WNLI** : 데이터셋 설계에서 문제가 있다는 말이 있어서, 실험에서 제외되었다.

 <br/>

### Appendix C : Additional Ablation Studies

#### C.1 Effect of Number of Training steps

 <img src="https://user-images.githubusercontent.com/11614046/109238425-dbf7dd00-7816-11eb-8c95-f1c27d59ad75.png" width="80%">  

 Figure 5는 MNLI Dev를 k번 사전학습을 한 뒤, 미세조정을 한 accuracy를 그린 표이다. 이를 보면 다음과 같은 질문에 이렇게 대답할 수 있다.

 - Q1 : BERT가 높은 finetuning accuracy를 달성하려면, 그렇게 많은 pretraining(128000words/batch * 1M steps)을 꼭 해야하나요?
 - A1 : 네. BERT base를 보면, 500k steps만큼 학습한 결과에 비해 1M steps 학습하면 1.0%의 성능향상이 있었습니다. 

 <br/>

 - Q2 : MLM이 모든 단어가 아닌 15%의 확률로만 학습을 하기 때문에 Left-to-Right model에 비해서 느리게 학습이 수렴되나요?
 - A2 : MLM은 LTR모델보다 수렴 속도가 느린 것은 맞습니다. 그러나, MLM의 성능은 LTR보다 즉시적으로 높은 성능을 보였습니다.

 <br/>
 
#### C.2 Ablation for Different Masking Procedures

 Section 3.1에서 우리는 BERT가 MLM 목적함수를 통해 사전학습을 하면서 토큰을 마스킹하는 여러 전략을 사용한다고 언급했다. 이번 파트에서는 다양한 마스킹 전략의 효과에 대해 알아보기 위한 ablation study를 해보도록 하겠다.  

 마스킹 전략의 목표는 미세조정에서 나오지 않는 [MASK] 토큰으로 인해 발생하는 사전학습과 미세조정 간의 차이를 줄이기 위한 전략이라는 점이다. MNLI와 NER에서 dev 데이터의 성능 결과는 Table8과 같다. NER에서는 fine-tuning과 feature-based방식을 둘 다 테스트해보았는데, feature-based는 representation을 재조정할 기회가 없기 때문에 불일치가 더욱 증폭될 것이라고 예상했다.  

 <img src="https://user-images.githubusercontent.com/11614046/109240452-e0be9000-781a-11eb-9658-08a6852378ec.png" width="80%">  

 표에 따르면, MASK는 타겟 토큰을 MLM을 하기 위해 [MASK]로 변환했음을 의미한다. SAME은 타겟토큰을 그대로 놔둔것이다. RND는 타겟토큰을 랜덤으로 다른 토큰으로 변환한 것을 의미한다.  
 
 표의 왼쪽 부분은 우리가 각 전략들을 어떤 확률로 사용했는지를 표현했다. (BERT의 경우 80% 10% 10%이다) 오른쪽은 Dev set의 성능을 보여준다. Feature-based 접근법에서는 Section 5.3에서 가장 좋은 성능을 보였던 것처럼 BERT의 마지막 4 layer의 값을 concat하였다.  

 놀라운 것은 fine-tuning이 어떤 masking전략을 사용해도 강건하다는 점이다. 그러나 기대했던대로, only Masking 전략은 feature-based방식으로 NER에 도전했을 때 성능이 저조했다. 흥미롭게도 RND만을 사용하는 전략도 우리의 방식보다 성능이 나빴다는 점이다.  

 
 <br/>


***
**개인적인 감상**

 18년 10월 발표된 BERT는 지금까지도 많은 사전학습 모델에 영향을 주고 있는 모델이다. XLNet, RoBERTa, ELECTRA 등 많은 BERT 계열의 모델들은 훌륭한 성능을 내지만 파라미터의 증가로 인해 학습시간이 무척 오래 걸린다는 문제점이 있다. 지금까지는 모델의 사이즈를 키우면서 자연어처리의 성능을 높여왔지만, 빠른 속도를 통해 실생활에 적용하기 위해서는 trimming 과정이 필요한 시점이 아닌가라는 생각이 든다.  
 
 또한, BERT는 일반적인 자연어 문제는 잘 해결하는 경향이 있지만, Finance Bio Chemistry 등의 특정한 분야에 적용하게 되면 성능이 급격하게 떨어지는 문제가 발생한다. 이에 따라, 특정 분야의 데이터만을 모아서 finance-BERT와 같은 모델을 만들고 있다.  
 
 Section 3.1, 5, Appendix가 이 논문에서 가장 중요한 부분인 것 같다. 논문을 읽을 때, 부록은 귀찮거나 별 내용이 없을 것이라고 생각해서 안 읽은 경우가 있는데 앞으로는 한번 체크해봐야 할 것 같다.  
 
 BERT 논문을 읽으면서 MLM, NSP와 같은 과제를 주어 unlabeled data로 학습을 진행한 것이 놀라웠다. 이후의 논문에서는 이런 학습전략을 개선하여 성능을 높인 것으로 알고 있는데, 어떤 식으로 개선해 나간 것인지 더 공부해보고 싶다.  
 
 <br/>

## Reference

https://mino-park7.github.io/nlp/2018/12/12/bert-논문정리/?fbclid=IwAR3S-8iLWEVG6FGUVxoYdwQyA-zG0GpOUzVEsFBd0ARFg4eFXqCyGLznu7w  


https://jeongukjae.github.io/posts/bert-review/  
