# 「From Light to Rich ERE : Annotation of Entites, Relations, and Events 」 Review  

![image](https://user-images.githubusercontent.com/11614046/176567676-c5c8d91a-3156-4fed-ab09-665ba79a1e1c.png) 

https://aclanthology.org/W15-0812.pdf

NAACL, 2015

<br/>

## 0. Abstract

우리는 DARPA DEFT 프로그램의 연구와 발전을 위해 만들어진 개체명, 관계, 사건 (ERE) 태깅과제의 발전을 설명한다. 우리는 DEFT의 문맥내에서 과제의 동기를 포함해서 Light ERE 태깅의 세부사항을 설명하는 것으로 시작한다. 우리는 Light ERE에서부터 더 복잡한 Rich ERE로 바뀌는 것을 설명하여, DEFT를 더 포괄적으로 이해할 수 있도록 한다.
 
<br/>

## 1. Introduction

 DARPA’s Deep Exploration and Filtering of Text(이하 DEFT) 프로그램은 추론, 일반적 관계, 이상탐지(DARPA, 2012)를 다루는 기술을 다루는데 초점을 맞춰서 자동화된 nlp의 sota 가능성을 높이는 것을 목표로 한다. DEFT프로그램내의 평가는 개체명과 사건 그리고 두개 간의 관계에 대한 정보를 이용하여 지식베이스를 풍성하게하는 문제에 일반적으로 집중하는 요소기술의 다양성에 집중한다. DEFT내의 다양한 접근방식과 평가들을 고려하면, 우리는 다양한 연구 방향성과 평가에 도움이 되며, 추론과 이상과 같은 좀 더 특수한 태깅과제에 유용한 근간을 제공할 수 있는 태깅과제를 정의하기 위해 이 일에 착수했다. ERE 태깅 과제의 결과는 프로그램에 따라서 발전해왔으며, 문맥 내의 경량화된 ERE에서부터 프로그램이 흥미를 갖는 현상의 풍부한 표현으로 발전해왔다.

 ACE, LCTL, OntoNotes, Machine Reading, TimeML, Penn Discourse Treebank, Rhetorical Structure Theory과 같은 이전의 접근 방식들은 이러한 과제 자원의 종류에 대한 백그라운드 작업의 일부를 만든 반면에, DEFT 프로그램은 기존의 존재하던 과제들의 정의를 넘어서 복잡하고 계층적인 이벤트 구조로 태깅하는 것을 요구한다. 다양한 언어와 장르에 대한 태깅 작업을 정의하는데 필요로하는 노력을 인식하여, 우리는 상당히 경량화된 구현부터 시작하여 시간이 지남에 따라 추가적인 복잡성을 도입하는 다단계 접근방식을 채택하기로 결정했다. 

 프로그램의 첫 단계에서, 우리는 다양한 언어로 라벨링된 데이터를 지속적으로 빠르게 생산할 수 있도록 하는 것을 목표로 Light ERE를 간단한 형태의 ACE 태그를 정의했다. 두번째 단계에서 Rich ERE는 ERE 온톨로지로 확장되며, 태깅할 수 있는 것들의 개념이 확장된다. 또한, Rich ERE는 이벤트 상호참조(특히 문서 내 및 문서 간 이벤트 멘션과 이벤트 아규먼트의 세분화 변화에 관련해서)의 어려움을 해소하기 위해 Event Hopper의 개념을 도입하였다. 그래서 (계층적이거나 중첩적인) 문서 간 이벤트 표현을 만드는 중요한 목표를 위한 길을 닦았다.

 남은 섹션에서 우리는 Light ERE 태깅 세부구분과 이 스펙 하에서 생성된 자원들에 대해서 설명한다. 우리는 Light ERE에서 Rich ERE로 전환한 동기를 자세히 설명하고, Rich ERE의 세부사항을 자세히 설명하면서 smart data selection과 annotation consistency analysis에 대해서도 다루겠다. 우리는 태깅의 어려움과 미래 방향성에 대한 논의로 결론짓도록 하겠다.
 
<br/>

## 2. Related Annotation Efforts

수많은 이전과 현재의 이벤트 태깅과제는 ACE와 TAC KBP로 평가하는 다양한 과제들을 포함하여 Rich ERE에 영향을 주었다. 우리는 각각을 아래의 항목들에서 차례대로 설명하겠다.

<br/>

### 2.1 ACE and Light ERE

 DEFT 프로그램이 시작했을 때, 시스템 트레이닝과 발전을 위해 빠르게 자원을 만들어서 스케일 업하는 것은 매우 중요했고, 그래서 우리는 우리의 지향점과 호환성이 있는 기존의 태깅 과제들을 찾아봤다. 그러한 과제 중 하나는 ACE(Automatic Content Extraction)으로 entity detection and tracking, relation detection and characterization 뿐만 아니라 event detection and characterization에 포커싱한 정보추출 벤치마크 과제이다. ACE 태깅은 people, organization, locations, geopolitical 개체명, weapons, vehicles 뿐만 아니라 각 개체타입의 서브타입까지 멘션들을 라벨링한 데이터이다. ACE는 또한 이러한 단어들 사이의 관계와 이벤트 셋을 태깅했다. 하나의 문서 내에서 복수의 멘션들이 같은 개체, 관계, 이벤트로 상호참조되었다.

 Light ERE는 태깅을 더 쉽고 일관적으로 하고자 하는 목적으로 ACE의 경량화 버전이면서 개체, 관계, 이벤트 태깅에 대한 간편한 접근 방식으로 디자인되었다. Light ERE는 더 적은 속성과 함께 감소된 개체와 관계 타입을 포착합니다. (예를 들어, 오직 특정한 개체와 실질적인 관계만이 태깅될 수 있으며, 개체의 서브타입은 라벨링되지 않는다) 이벤트들은 ACE와 Machine Reading에서 개발된 방법론을 따라서 라벨링되지만, Discussion Forum(DF)와 같은 비공식적인 장르들에 맞게 조정된다. Light ERE의 이벤트 온톨로지는 ACE의 이벤트 온톨로지를 약간 수정하고 줄였기 때문에 Ace의 온톨로지와 유사하며, 이벤트들은 문서들 내에서 상호참조된다. ACE에서와 같이, 각 이벤트 멘션의 태깅은 트리거의 세부사항, 이벤트 타입과 서브타입의 라벨링, 이벤트 아규먼트 개체를 포함한다. ACE를 간소화해서, 오직 확인된 실제 이벤트만이 태깅된다. (비현실적인 이벤트와 아규먼트는 제외된다)

 우리의 Light ERE에의 노력은  영어뿐만 아니라 중국어와 스페인어로 된 풀 태깅 자원을 만들어서 cross lingual이 될 수 있도록 부분적인 태깅데이터를 만들었다는 점도 포함된다. 우리는 10만개의 중국어 단어로 이루어진 데이터에 영어 번역이 추가된 중국어-영어 Light ERE 코퍼스를 만들었다. 이 평행 데이터의 일부분들은 중국어는 Chinese Treebank, 영어는 English Treebank처럼 수행된 언어에 따라 다른 태깅 레이어를 갖는다. 동일한 데이터셋에 맞춰진 ERE, treebank와 같은 다중 레벨 태깅은 다중레벨을 함께 조작하는 머신러닝 방법론 실험을 용이하게 할 것으로 예상되는 자원과 함께 제공되어야 한다.

<br/>

### 2.2 TAC KBP Event Evaluations

 Text Analysis Conference(TAC)는 NLP연구를 장려하고 대용량 데이터셋, 일반적인 평가 절차, 연구자들이 그들의 연구결과를 공유할 포럼을 개발하는 Natural Institute of Standards and Technology(NIST)에 의해 조직된 워크샵이다. 다양한 평가들을 통해서, TAC의 Knowledge Base Population(KBP) 트랙은 Knowledge Base에 나타나는 자연어 본문들에서 언급된 개체들을 연결하고, 문서에서 개체에 대한 새로운 정보를 추출하고, 이를 Knowledge base에 추가하는 시스템의 발전을 장려한다.

 2014년, TAC KBP는 시스템이 비구조적 텍스트에서 개체명의 멘션을 추출하고 텍스트 내에서 하는 역할을 찾는 시스템을 요구하는 Event Argument Extraction(EAE)의 평가가 추가된 이벤트 도메인으로 이동했다. 추가적으로, 2014 TAC KBP는 또한 이벤트 트리거, 타입과 서브타입 분류, 속성 정보로 구성되는 event nugget triple을 찾는 시스템을 요구하는 Event Nugget Detection(END)에 대한 시범 평가를 수행했다.

 2015년 TAC KBP의 ENA와 END는 둘 다 동일한 이벤트에 참가하여 event argument를 연결하거나(EAE) 같은 이벤트를 언급하는 이벤트 너겟을 그룹화하여(END) 이벤트 튜플들이 함께 그룹화되거나 서로 연결되도록 확장할 계획이다. 이러한 확장은 ACE와 Light ERE에서 도전과제로 삼는 이벤트 상호참조 구분을 필요로하는 평가과제이다. Light ERE에서 Rich ERE로의 전이는 event hoppers를 추가하여 이 과제를 해결합니다.

<br/>

## 3. Transition form Light ERE to Rich ERE

Attention으로 대체된 encoder-decoder구조는 다음과 같다. 




### 3.1. Encoder

인코더은 bi-RNN구조로 이뤄져 있다. 
 
<br/>

위의 그림과 같이 정방향 rnn으로 fw_h_i들을 만들고, 역방향 rnn으로 bw_h_i를 만든다.

해당 fw_h_i와 bw_h_i를 concat해주면, 해당 input값인 x_i에 대한 hidden_state를 구한 것이다. 

기존의 인코더와는 다르게 S라는 fixed_size vector를 구하지 않는다.

<br/>

### 3.2. Decoder

디코더는 기존의 디코더와 크게 다르지 않다.

기존 디코더는 현재 단어(y_i)를 알기 위해, 이전의 단어(y_i-1) + 이전의 state(s_i-1)를 이용해 현재의 state(s_i)를 구했다. 이렇게 구한 s_i로 y_i를 찾아낸다.

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