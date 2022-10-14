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

Light ERE의 간소화된 태그는 태그를 빠르게 확장할 수 있도록 해주었다.  DEFT 프로그램이 더 복잡한 알고리즘과 평가로 이동함에 따라, ERE 프레임워크 내에서 더 풍부한 이벤트의 표현으로 전환하는 것은 필수적이게 되었다. Rich ERE의 개발은 문사 간의 교차, 언어 간의 교차 이벤트 표현 뿐만 아니라 이벤트 - 이벤트 관계의 영역으로 향후 확장을 위한 토대를 마련한다. Rich ERE로의 전환은 확장된 이벤트와 이벤트 구문에 대한 태깅 가이드라인의 발전과 새로운 태깅 과제를 다룰 새로운 태깅툴의 발전을 요구한다. 

<br/>

### 3.1. Development of Annotation Guidelines for Rich ERE

#### 3.1.1 Expanded Entity Annotation

 Rich 엔티티 태깅은 일반적으로 태깅가능성의 증가부터 시작하여 Light 태깅의 많은 영역을 확장한다. 태깅을 세부적이고 지정된 엔티티로 제한하는 대신에, 우리는 ACE가 비세부적이며 일반적인 엔티티라고 부르는 대상까지 Rich ERE 태깅의 범위에 추가하였다. “non-specific(NonSPC)”라는 포괄적 용어 하에서, 우리는 Light ERE에서 수집하는 Specific(SPC) 엔티티를 포함해서 비세부적이며 일반적인 엔티티까지 이제 수집한다. 우리는 Light ERE 데이터를 태깅하는 도중에 일반적인 언어를 포함한 많은 토론 포럼 문서를 접했다.  이전에 우리는 이러한 문서들을 제외했지만, Rich ERE에서는 NonSPC 엔티티를 포함했기 때문에 우리의 태깅가능한 문서의 범위는 더욱 확장됐다.

 ACE의 특징 중 우리가 되살린 몇가지들은 명목적인 헤드마킹과 Location과 Facility 개체타입을 구분한 것이다. ACE에서 요구하는 대로 명명되었거나 수식하는 헤드를 마킹하는 대신에, 헤드들은 Rich ERE에 있는 수식 멘션들만 수동적으로 마킹되었다. 명명되었거나 수식하는 헤드들이 일반적으로 엔티티 멘션과 같은 텍스트이기 때문에, 그들의 헤드들은 수동적으로 따로 마킹될 필요는 없다. 그러나, 명목상의 멘션 헤드들은 사소한 파생이 아니기 때문에, 그들은 Rich ERE에서 수동적으로 마킹되었다. 게다가, Light ERE는 지역, 지형, 건물, 그리고 다른 구조물들을 Location 엔티티 타입에 통합시켰다. ACE에 이어서 TAC KBP 평가과제를 더 잘 일치시키기 위해서, Rich ERE는 Light ERE의 Location 엔티티 타입을 Facility와 Location 타입으로 분리하였다. 인공 건축물과 기반 시설들은 Facilities로 간주되었으며, 지역, 지형, 그리고 다른 비기술 지역들은 Location으로 나뉘어집니다. 예를 들면 아래와 같습니다. (명목상의 멘션의 헤드는 밑줄로 표현됩니다.) 

 - [Tourists]PER.NOM.NonSPC always end up at [Love Park]FAC.NAM.SPC

 - [The last four tourists to show up]PER.NOM.SPC missed the bus

 게다가, 우리는 엔티티 수준에서 태깅되지 않는 이벤트와 관계를 체크하는 Fillers 구문의 클래스를 새로 만들었다. Filler 구문은 태그된 관계나 이벤트에서 구문 역할을 충족시킬 때만 태깅이 된다. Filler 구문의 예시는 아래의 관계와 이벤트의 논의에 포함되어 있다. ACE는 weapon과 vehicle을 엔티티로 철저하게 태그한 반면에, Rich ERE는 그것들을 argument filler로 태깅하였다. Rich ERE는 또한 commodities 태그를 filler로 태깅하였다.

 추가적으로, Light ERE에서 제목 엔티티는 argument filler로 재분류되었는데, 왜냐하면 이들이 관계 구문에서 명명된 사람과 연결될 때만 태깅이 되었기 때문이다. argument filler로 분류된 것들은 Title, Age, URL, Sentence, Crime, Money, Vehicle, Weapon, Commodity, Time 타입이다. 이러한 각 요소들은 특정한 관계나 이벤트 서브타입에 대응되며, 이는 상응하는 서브타입이 해당 정보에 적합할 경우에만 표시된다는 의미이다. 예를 들어, 한 사람의 나이는 일반적인 소속-인격 관계의 argument filler로만 태깅될 수 있으며, weapon은 conflict, attack, manufacture.Artifact, Life.injure와 같은 제한된 숫자의 이벤트 서브타입에서만 태깅이 된다.

<br/>

#### 3.1.2 Expanded Relation Annotation

 Rich ERE 관계는 Light ERE의 10개 서브타입에서 Rich ERE의 20개 서브타입으로 온톨로지를 2배 늘림으로써 영감을 얻기위해 TAC KBP Slot Filling 평가를 참고하였다. KBP Slot Filling 과제는 태깅하는 사람에게 ERE 태깅과 범위가 매우 유사한 텍스트 정보를 찾도록 요구합니다. 예를 들어, ERE와 KBP Slot Filling는 둘 다 자회사-모회사 관계와 조직의 위치뿐만 아니라 조직, 지분관계, 국적에 기반한 요소내에서 개인의 고용과 멤버쉽에 기반한 자료들에 대해 태깅합니다. 이는 KBP Slot Filling의 더 많은 측면을 ERE 관계 온톨로지에 통합하기 위한 자연스러운 과정입니다. 크로스-프로젝트 동기화의 일부분은 어떤 관계 타입에 대해서 새로운 argument filler가 필요합니다. 관계에서 3개의 새로운 서브타입은 아래에 기술된 argument filler를 사용합니다. 개인-사회적 역할(제목), 일반 소속 또는 웹사이트(URL), 일반소속 또는 인물(나이). 표1은 Light ERE와 비교하여 Rich ERE에 새롭게 추가된 관계 목록이다. 

 <img src="https://user-images.githubusercontent.com/11614046/177472065-5c2b2f38-cf3b-4a7f-82e5-75a68623971a.png" width="70%">

 마침내, Light ERE는 증명되고 확정된 관계에만 태깅을 하였지만, Rich ERE는 미래, 가설 그리고 조건부(부정되지는 않은) 관계에도 태깅을 하였다. 모든 관계들은 '확정된 것' VS '그 외 기타'의 차이점을 구별하기 위해 실재적인 속성에 할당된다. 이 추가점들과 변화점들은 아래의 예시에서 보여진다. 

 - Now [53]AGE.ARG, [Barack Obama]PER.NAM.SPC signed important documents this morning. (General-Affiliation.PER-Age, Realis: Asserted)

 - [[Spanish]GPE.NAM.SPC students]PER.NOM.SPC gathered to protest the growing cost of tuition. (General-Affiliation.MORE, Realis: Asserted)

 - [She]PER.PRO.SPC has been living in [California]GPE.NAM.SPC for three years now. (Physical.Resident, Realis: Asserted)

 - [He]PER.PRO.SPC may end up in [New York]GPE.NAM.SPC. (Physical.Located-Near, Realis: Other)

<br/>

#### 3.1.3 Expanded Event Annotation

 각각의 이벤트 멘션에 대해서, Rich ERE는 이벤트 타입과 서브타입, 그것의 실재 속성, 존재하는 구문 또는 참여자, 텍스트에서 요구되는 트리거에 대해서 레이블링하였다. 

 Rich ERE의 이벤트 태깅은 Light ERE의 이벤트 태깅에 비해서 여러 분야에서의 태깅가능성이 높아진다. (약간 증가된 이벤트 온톨로지, 일반 및 기타(비실재) 이벤트 멘션의 추가, 이벤트 멘션에 대한 argument 없는 트리거의 추가, contact와 transaction 이벤트에 관한 추가적인 특성들, 멀티플 타입/서브타입에 대한 이벤트 태깅의 이중태그, 그리고 특성 타입의 coordination에 대한 이벤트 멘션의 다중 태깅)


<br/>

 **A. Expansion of event ontology, and additional attributes for Contact and Transaction events**

 Rich ERE는 Light ERE의 이벤트 타입 목록에서 하나의 새로운 이벤트 타입을 추가합니다. 이벤트 타입의 전체 목록은 다음과 같습니다 : Life, Movement, Business, Conflict, Contact, Personnel, Transaction, Justice, Manufacture. Manufacture 이벤트 타입에는 Manufacture.Artifact라는 하나의 서브타입만이 있으며, 다음과 같은 argument를 가집니다 : agent, patient(weapon, facility, vehicle, commodity), time, location

 예를 들면,

 - [China]AGENT is reportedly **constructing** [a second aircraft carrier]PATIENT.VEHICLE

 - [the Imboulou hydroelectric power station]PATIENT.FACILITY, which was **constructed** by [Chinese technicians]AGENT

 새로운 이벤트 타입 외에도, Rich ERE는 이미 존재하는 이벤트 타입에 몇몇 새로운 서브타입을 추가하였다. : Movement.Transport-Artifact, Contact.Broadcast, Contact.Contact, Transaction.Transaction.

 Movement.Transport-Artifact 서브타입은 weapon, vehicle, facility, commodity를 patient로 가질 수 있습니다. 예를 들면,

 - [122 kilos of heroin hidden in a truck]ARTIFACT.COMMODITY which was set to **cross** into [Greece]DESTINATION.GPE 

 - [the cans of marijuana]ARTIFACT.COMMODITY were **launched** about 500 feet into the [U.S.]DESTINATION.GPE using [a pneumatic powered cannon]INSTRUMENT.WEAPON 

 Contact 이벤트 멘션은 Formality (Formal, Informal, Can’t Tell), Scheduling (Planned, Spontaneous, Can’t Tell), Medium (In-person, Not-in-person, Can’t Tell), Audience (Two-way, One-way, Can’t Tell)의 속성으로 레이블링됩니다. Contact 이벤트 서브타입은 태깅된 속성에 따라 자동으로 결정된다. 

 - Contact.Meet: Medium attribute must be “In-person” and audience attribute must be “Two-way” 

 - Contact.Correspondence: Medium attribute must be “Not-in-person” and audience at- tribute must be “Two-way” 

 - Contact.Broadcast: Any Contact event mention where the audience attribute is “One-way”

 - Contact.Contact: Used when no more spe- cific subtype is available, and occurs when either the medium or audience attribute is “Can’t Tell”

 Contact.Meet 와 Contact.Correspondence 는 Light ERE의 서브타입에서 변하지 않았지만,  Contact.Broadcast 와 Contact.Contact 는 Rich ERE에 추가된 새로운 서브타입이다.

 Formality 와 Scheduling 속성은 모든 Contact 멘션에 태깅이 되지만, 이러한 속성은 서브타입 결정에 영향을 미치지 않는 점을 명심해야 한다. 

 Transaction.Transaction 은 transaction이라는 이벤트가 언급이 된 것은 확실하지만, 문맥상 돈이나 상품이 전달되었는지는 명확하지 않은 경우를 나타내기 위해 추가된 새로운 서브타입이다. 예를 들어,

 - I **received** a gift (Transaction.Transaction)

 <br/>

 **B. Addition of generic and other irrealis event mentions**

 ERE 태그를 현재의 EAE와 END 과제와 더 가깝게 정렬하기 위해서, Rich ERE는 각 이벤트 멘션에 Realis 속성을 부여했다. 이는 EAE와 END 모두와 동기화되며, ACE 태깅과도 호환이 된다.

 Realis 속성은 Actual(asserted), Generic(generic, habitual), 기타(future, hypothetical, negated, uncertain)이 있다. 이전의 Light ERE 태그는 Actual Event 멘션으로만 제한되었다.

 - Actual: He **emailed** her about their plans
 - Other: Saudi Arabia is scheduled to begin **building** the world’s tallest tower next week
 - Generic: Turkey is a popular passageway for drug smugglers **trafficking** from south Asia
to Europe
 
 이벤트 멘션의 realis와는 별개로 argument와 이벤트 멘션간의 realis 관계 역시 태깅될 것이다. 예를 들면,

 - [+irrealis] “Jon” as the agent for the asserted Conflict.Attack event: [Jon] denied [he] master-minded the **attack**

 <br/>

 **C. Addition of argumentless triggers for event mentions**

 Light ERE와 다르게, Rich ERE는 텍스트에 있는 이벤트의 참가자나 argument가 존재하지 않더라도 이벤트 멘션 트리거의 태깅을 허용한다. 이 추가적인 태깅은 Rich ERE가 END와 더욱 유사하게 만들어준다.

 <br/>


 **D. Double tagging of event mentions for multiple types/subtypes**

 Rich ERE는 ERE 이벤트 분류법에 있는 필수 추론 이벤트를 태그할 수 있도록 이벤트 트리거의 이중 태깅을 허용합니다. 예를 들어, 만약 Transaction 이벤트로 money와 ownership이 둘 다 이전된다면, 이벤트 멘션은 각각의 서브타입으로 한번 씩 두번 태깅되어야 할 것이다.

 - I **paid** $7 for the book (tagged as both Transaction.TRANSFER-OWNERSHIP, and Transaction.TRANSFER-MONEY)

 이러한 방식으로 태깅된 트리거들은 문맥상에서 하나 이상의 이벤트/서브타입을 명확히 나타내는 트리거들로 제한된다.

 - Conflict.Attack and either Life.Injure or Life.Die : murder, victim, decapitate, kill
 - Transaction.Transfer‐Money and Transaction.Transfer‐Ownership (money being exchanged for an item): buy, purchase, pick up
 - Legal language that might trigger multiple Justice Events or other Event Types: guilty plea, execution (Life.Die / Justice. Execute), death penalty, testimony (Justice.Trial Hearing, Contact.Meet)

 Light ERE로부터의 변경에서 이벤트 트리거들은 엔티티와 동일한 텍스트이거나 NOM 엔티티 멘션의 헨드와 동일한 문자열일 수도 있다. 엔티티 멘션 내에서 중첩된 이벤트 트리거 역시 허용된다.

 - The situation escalated and the **[murderer]** fled the scene. (This is an event trigger, even though “murderer” would already be a nom- inal PER entity.)
 - The mayor agreed to meet with **[angry protestors]**. (This is a trigger, even though “protesters” would already be the head of a nominal PER entity.)
 - **[The one who divorced me]** only thinks of himself. (Here “divorce” can be a trigger for a Life.DIVORCE event, even though it is nested within a longer PER entity and it is not the head noun.)

 <br/>

 **E. Multiple tagging of event mentions for certain types of coordination**

 Rich ERE는 또한 argument의 조정을 통해 여러 이벤트가 표시된 경우, 단일 트리거에 여러 번 태그를 진행할 수 있다. 조정된 Argument의 역할은 단일 이벤트 멘션이나 여러 이벤트 멘션에 태깅이 되는지의 여부를 결정하는 것이다.

 - 만약, Time이나 Place의 역할이 조정되거나, 별도로 표시된 Time과 Place가 있는 경우에는 여러 이벤트가 태깅됩니다. 
 - 만약 다른 argument의 역할이 조정된다면, 단일 이벤트가 태깅된다. 이런 경우에, 각각의 조정된 argument들은 이벤트 멘션의 argument로 따로 태깅되며, 결과는 단일 이벤트에 복수개의 멘션이 조정된 argument 역할에 태그가 지정된 복수의 argument으로 구성된 단일 이벤트이다.

 만약 문맥이나 언어가 너무 복잡해서 이벤트의 개수를 알기 어렵다면, 태깅자는 여러개의 argument로 하나의 이벤트에 태깅하도록 교육받았다.

 이 예시를 보면, Time argument가 다르기 때문에 2개의 Conflict.Attack 이벤트가 존재하며, “murderer”에 의해 발생된 2개의 Life.Die 이벤트가 존재한다.

 - Cipriani was sentenced to life in prison for the **murder** of Renault chief George Besse in 1986 and the head of government arms sales Rene Audran a year earlier
   - Conflict.Attack: Trigger = murder, agent = Cipriani, victim = George Besse, time = 1986
   - Conflict.Attack: Trigger = murder, agent = Cipriani, victim = Rene Audran, time = a year earlier
   - Life.Die: Trigger = murder, argument = George Besse, agent = Cipriani, time = 1986
   - Life.Die: Trigger = murder, argument = Rene Audran, agent = Cipriani, time = a year earlier

 아래의 예제에서는 복수 개의 giver argument와 복수 개의 recipient argument가 있지만 1개의 이벤트만이 태깅되었다.

 - China and the US are the biggest **lenders** to Brazil and India
   - Transaction.Transfer-Money: Trigger = lenders, giver = China, giver = US, recipient = Brazil, recipient = India

 <br/>

#### 3.1.4 Event Hoppers and Event Coreference

 ACE뿐만 아니라 Light ERE에서 이벤트의 상호참조는 이벤트 정의를 엄격하게 하기 위해서 제한되었다. 구성요소기준에 따라서 Light ERE에서 태깅자들은 같은 사람, 대상, 시간, 위치라면 두 사건을 상호참조 사건으로 태깅하였다, 그러나, 태깅자들이 직관적으로 느끼기 때문에 엄격한 이벤트 분류 기준을 맞추지 못한 이벤트 멘션들이 많았으며, Light ERE와 ACE에서 상호참조되지 않은 것들이 많았다. 몇몇 이벤트들은 태깅자들의 직관적인 판단과 엄격한 상호참조 기준 사이의 헷갈림때문에 상호참조로 일관되게 태깅되지 않았다. 

 Rich ERE에서는 Event Hopper의 개념을 더 포괄적이고 이벤트 상호참조의 개념으로 만들었다. Event hopper는 이전이라면 이벤트 요구조건을 충족시키지 못했을 이벤트이더라도 태깅자들이 상호참조라는 느낌을 주는 이벤트에의 언급을 포함한다. 좀 더 구체적으로는, 동일한 hopper에 속하는 이벤트 멘션들의 특징은 다음과 같습니다.

 - 동일한 이벤트와 서브이벤트를 갖습니다. (Contact.Contact와 Transaction.Transaction 멘션은 각각 어느 Contact와 Transaction에 추가될 수 있기 때문에 예외입니다.)
 - 동일한 시간, 장소 범위이다. 완벽하게 똑같은 날이거나 같은 시간적 표현일 필요는 없다. (Attack in Baghdad on Thursday vs. Bombing in the Green Zone last week)
 - 트리거의 형태는 다를 수 있다. (Assaulting 32 people vs wielded a knife)
 - 이벤트 argument는 상호호환적이지 않거나 충돌할 수 있다 (18killed vs dozens killed)
 - Realis 상태는 다를 수 있다. (will travel [OTHER] to Europe next week vs. is on a 5 day trip [ACTUAL])

 태깅된 모든 이벤트 멘션들은 Rich ERE에서는 모두 event hopper로 포함되며, 동일한 이벤트 발생을 나타내는 모든 태그된 이벤트 멘션들은 동일한 event hopper로 그룹화된다.

 Event hoppers는 태깅자들이 더 많은 멘션들을 그룹화할 수 있도록 해줘서, Rich ERE에 더 많은 이벤트 argument가 라벨링되도록 도와준다. 이렇게 더 풍부한 태깅은 더 완벽한 지식베이스와 2015년의 Event Argument Linking and END 평가에서 더 나은 지원으로 이어질 것이다. 

 <br/>

### 3.2 Development of an Annotation GUI for Rich ERE

 Rich ERE 태깅도구는 Wright et al.(2012)에 설명된 프레임워크를 따라서 개발되어, Rich ERE를 위한 새로운 인터페이스를 빠르게 개발할 수 있게 되었다. 많은 기능들이 이전 인터페이스를 위해 개발되었기 때문에 추가적인 개발 시간이 필요하지 않았다는 점에서 무료로 포함되었다. 이 중에 하나의 중요한 예시는 임의적으로 겹칠 수 있으며 다른 태그(예를 들면 엔티티 타입)들을 색칠할 수 있으며, 사용자들이 태그들 사이에서 클릭하여 탐색할 수 있는 기능이 가능한 텍스트 범위를 태깅하는 방식이다. Rich ERE를 위해 특별히 개발된 중요한 기능은 서로를 가르킬 수 있게 해주는 “참조 태깅” 기능이다. 하나의 멘션이나 개체명에 대한 완전한 태깅이 수행되면, 단일 태깅은 관계나 이벤트 argument에 연결이 될 수 있지만, 그러나 참조하면 기존의 태깅값들이 안전하게 변경될 수 있습니다. 또한, 태깅 매니저들은 인터페이스가 정의된 데이터베이스에의 직접적인 접근권한이 허용된 편집자이기 때문에, 태깅툴의 세부사항을 넘어서 개발의 큰 역할을 했다. 매니저들은 위젯을 추가하고, 변경하며(예를들 면, 메뉴 선택 추가), 태그들간의 논리적 제약(예를 들면, “resident”관계는 person argument를 가져야만 한다)을 정하기도 하였다.

 <br/>

## 4. Linguistic Resources Labeled for ERE

 현재까지 우리는 NW와 DF를 포함하여 약 57만개의 영어 Light ERE 데이터와 20만개의 중국어 DF를 공개했다. 10만개의 스페인어 Light ERE는 현재 진행 중이며 몇 주 안에 완성될 것으로 예상된다. 영어 Rich ERE 태깅은 현재 진행중이며, 현재까지 32420개의 단어(91개문서)가 완료되었다. 우리는 17만개의 영어단어와 10만개의 중국어 스페인어 단어를 몇 주 안에 완성할 것으로 예상한다. Rich ERE 데이터의 일부분은 새롭지만, 나머지는 이전에 Light ERE에서 태깅되었던 것들이다. 각 언어, 장르, 과제의 세부사항은 아래의 Table 2에 나와있다. ERE 데이터는 현재 DEFT와 TAC KBP 수행자가 이용할 수 있으며, 미래에는 LDC의 카탈로그에도 게시되어 대부분의 연구자 커뮤니티가 활용할 수 있도록 할 것이다.

 <img src="https://user-images.githubusercontent.com/11614046/179447437-916a2544-bb7e-49ac-8148-06834ca709e5.png" width="60%">

 DEFT의 이번 단계의 대부분의 타겟은 영어, 중국어, 스페인어 데이터에서 각 언어당 400Kw의 Rich ERE를 완성하는 것이다. 스페인어와 중국어의 100Kw는 같은 데이터의 영어 번역에 대한 Rich ERE와 상응하도록 만들 것이다. 우리는 이 태깅 목표가 이번해 말까지 달성될 것으로 기대한다.

 <br/>

### 4.1 Smart Data Selection

 내용이 부족한 문서에 대한 태깅자의 노력을 최소화하기 위해서, 문서들은 1000토큰당 이벤트 트리거의 개수를 이용한 문서 트리거 밀도의 내림차순으로 태깅 파이프라인에 입력되었다. 트리거들은 자동적으로 ACE 2005 태깅(Walker et al., 2006)에 대해 훈련된 심층 신경망기반 태그와 맞춤 및 단어임베딩 기능을 활용하여 자동으로 태그를 지정하였다. 단어임베딩은 word2vec을 활용하여 수십억 개의 뉴스와 포럼 데이터의 수십억개의 단어를 훈련된 임베딩값이다. 이 선택 과정을 활용한 예비 결과는 태깅자들이 이전의 순서가 없었던 방식에 비해서 평균적으로 훨씬 풍부한 문서를 보고할 수 있었다는 점에서 고무적이었다.

 <br/>

### 4.2 Rich ERE Challenges and Next Steps

 이벤트 태깅에서의 한가지 과제는 서브이벤트 vs 이벤트 호퍼 를 구별한 기준을 정하는 것이다. 우리는 테스트 Rich ERE 태깅에서 이러한 문제를 발견하였으며, 목표는 서브이벤트가 미래에 이벤트 호퍼간의 관계를 갖도록 하는 것이다. 이벤트 호퍼사이의 관계를 표현하기 위해서, 우리는 인과성, 부분-전체, 전례, 가능성 등과 같은 비정체성 이벤트-이벤트 관계를 포착하기 위해서 Narrative Container (Pustejovsky and Stubbs, 2011)와 같은 개념을 추가할 계획이다. 이벤트 호퍼는 개별 이벤트 멘션과 Narrative Containers 사이의 수준을 정하는 역할을 한다. 이벤트 호퍼는 Narrative Containers로 그룹화되며, 관계들은 개별 이벤트 멘션이 아닌 이벤트 호퍼간의 관계가 되게 된다. 개별 이벤트 멘션간의 더 자세한 관계들은 Narrative Containers 간의 관계나 Narrative Containers의 이벤트 호퍼 사이의 이벤트-이벤트 관계에서 도출된다.

 <br/>

### 4.3 Inter-Annotator Agreement

 주석자간의 합의(IAA)에 대한 작업은 엔티티 멘션에서부터 이벤트까지 태깅 계층의 각 레벨에 사용되는 알고리즘을 설명하는Kulick et el(2014)의 방식을 기반으로 한다. 이 작업은 전체 엔티티뿐만 아니라 엔티티, 관계, 이벤트 멘션에 대한 평가까지도 초점을 둔다. 엔티티 멘션 맵핑의 알고리즘은 엔티티 멘션의 범위를 기반으로 하는 반면에, 관계와 이벤트 멘션의 맵핑은 엔티티 멘션 매핑에 따라 달라지는 argument들의 맵핑을 기반으로 하기 때문에 더욱 복잡하다. IAA 작업은 Rich ERE에 대한 이중 태깅에 대해 수행될 것이다. 분석 결과는 미래에 발표할 예정이다.

 <br/>

## 5. Conclusion 

 Rich ERE 태깅은 확장된 태그 가능성, 확장된 범주, 현실 및 특수성에 대한 태깅, 이벤트 호퍼 레벨에서의 확장된 상호참조성을 포함하여 엔티티, 관계, 이벤트 태깅의 확장성도 포함한다. 확장 및 변경은 지식베이스에의 정보를 더욱 확장시켜줄 것이다. 미래에는 Rich ERE의 확장, 구체적으로 말하면 태그가능성의 확장과 이벤트 호퍼 레벨의 느슨한 상호참조성은 문서내의 이벤트-이벤트 관계와 궁극적으로 문서간과 언어를 넘어선 태깅의 지원까지 향상될 것으로 기대된다.

  

 <br/>

## 6. 요약

