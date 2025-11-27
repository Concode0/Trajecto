# Ground Truth의 딜레마: iPad는 믿을 수 있는가? + 보간법

본 Trjaeco 프로젝트의 EKF-TCN 모델 성능(ATE, RPE)은 전적으로 '정답' 데이터인 Ground Truth(GT)의 신뢰도에 의존한다. 만약 GT 자체가 부정확하다면, 모델의 모든 성능 지표는 무의미해진다.

따라서, iPad Air와 Apple Pencil Pro 조합을 GT로 사용하기에 앞서, (1) 이 시스템이 학술적으로 신뢰할 수 있는지, (2) 내재적 한계는 무엇이며, (3) 그 한계를 극복할 수 있는지에 대한 엄밀한 검증이 선행되어야 했다.

## 신뢰도 핵심 요소 : 데이터의 속도와 깊이

iPad의 디지타이저는 Apple Pencil의 입력을 초당 240회의 속도로 업데이트 할 수 있습니다. (Coalesced Touches) 어플리케이션 설계 시에 화면의 업데이트 빈도에 의해서 데이터 획득 주기가 맞춰지는 걸 방지하기 위해서 별도의 이벤트를 이용해서 240Hz 샘플링이 가능하도록 설계했습니다. 

또한 Apple Pencil Pro를 통해서 단순한 2D 위치뿐만이 아니라, 압력, 기울기, 회전, 호버링에 대한 데이터들을 얻을 수 있으므로 풍부한 동적 데이터를 얻을 수 있음을 확인했습니다.

## 실증적 검증 : 연구에서의 활용 사례

Trajecto 프로젝트에서 실궤적 데이터로서의 사용하기 위해서 다른 분야의 연구에서 이미 검증된 사실들을 확인했습니다. 조사 결과 여러 고정밀 연구 분야에서 검증되었습니다. 

1. 생체 인증: 한 연구에서는 Apple Pencil에서 수집된 필기 동역학(압력, 방위각, 고도, 위치 등 5가지 특징)을 사용하여 사용자를 인증하는 분류 모델을 훈련했습니다. 그 결과, 다수 사례에서 99% 이상의 높은 인증 정확도를 달성했습니다. 이는 수집된 데이터가 개인의 고유한 특징을 일관되게 포착할 만큼 정밀하고 신뢰할 수 있음을 입증합니다. [https://www.ndss-symposium.org/wp-content/uploads/usec2024-56-paper.pdf](https://www.ndss-symposium.org/wp-content/uploads/usec2024-56-paper.pdf)
2. 임상 진단 (디지털 바이오마커): 여러 의학 연구에서 iPad를 사용하여 전통적인 '시계 그리기 검사(CDT)' 또는 '레이-오스터리츠 복합 도형 검사(RCFT)'를 디지털화했습니다.29 이 시스템은 종이와 펜으로는 불가능했던 운동학적 데이터(예: 총 획 수, 펜이 표면에 닿아있는 시간, 공중에 떠 있는 시간)를 캡처할 수 있습니다.30 이러한 '디지털 바이오마커'는 경도인지장애(MCI) 환자 그룹과 건강한 노인 그룹을 구별하는 데 유의미한 차이를 보였습니다. [https://mhealth.jmir.org/2024/1/e48777/PDF](https://mhealth.jmir.org/2024/1/e48777/PDF)
3. 발달 장애 스크리닝: 아동의 쓰기 장애(dysgraphia)를 조기에 선별하는 연구에서도 iPad와 Apple Pencil이 활용되었습니다.2 연구팀은 펜 스트로크의 운동학적 파라미터(속도, 압력 등)를 분석하여, 기존 방식보다 2년 더 이른 시기에 84.62%의 정확도로 '위험군' 아동을 식별해냈습니다. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10054332/pdf/life-13-00598.pdf](https://pmc.ncbi.nlm.nih.gov/articles/PMC10054332/pdf/life-13-00598.pdf)

## 한계점 및 고려사항

iPad Air와 Apple Pencil Pro 가 높은 정확도를 가짐을 확인했음에도 불구하고 정밀 측정 장비가 아니기 때문에 내재적 한계가 있음을 확인할 수 있었습니다.

디지타이저의 센서 정확도의 구조적 문제로 인해서 가장자리와 모서리에서 정확도가 저하되는 문제가 있었습니다. 그러나 이러한 문제 해결을 위해서 데이터 획득 시에는 최대한 중앙지점에 가깝도록 작성되도록 했습니다.

또한 호버링 데이터는 EMR 신호를 기반으로 구현되기 때문에, 최대 12mm 거리까지 펜을 감지할 수 있습니다. 그러나 데이터를 오직 정규화된 데이터만을 제공하기 때문에 이를 역산해서 정확한 거리를 구하는데 다른 방안을 고려해야했습니다.

그리고 호버링 데이터는 iPad Air와 Apple Pencil Pro 의 내장 한계로 인해서 호버링 시에 데이터가 60Hz로 획득되는 문제를 고려해야 했습니다.

## zOffset 보정 과정

### 1. 선형 가정

$$
Distance = 12 * zOffset
$$

정규화된 Offset을 선형화되어있다고 가정해서 최소를 0, 최대를 12mm로 설정해서 역산하는 방식을 처음 생각했습니다. 그러나 이러한 방식은 EMR 방식의 원리와 맞지 않아서 호버링 거리의 중간 정도 지점에서 오차가 극대화될 수 있는 가능성이 있고 정확도의 저하가 우려됩니다.

### 2. 물리 기반 지수 모델

EMR을 통한 거리 획득 과정에서의 구조적인 원리를 고려하면 신호 강도가 거리에 제곱으로 반비례함을 가정할 수 있다. 이를 이용해서 아래와 같이 거리 식을 구성할 수 있습니다.

$$
\text{Distance}(\text{mm}) = A \cdot (zOffset)^p
$$

여기서 A는 스케일링 팩터, p는 멱지수이다. 이론적으로는 0.5에 근접할 것으로 기대하며, A는 최대거리인 12mm로 값을 기대할 수 있습니다. 이러한 방식은 물리적으로 가장 설명력 있고 강건한 호버링 거리 재구성을 할 수 있을 것이라고 판단했습니다.

## 데이터 샘플링 보간 방식

### 1. 선형 보간

궤적에 '각진' 지점을 만들어 가짜 가속도를 유발한다. 이는 '가속도 기반 DTW 정렬'이나 'TCN 훈련'에 치명적인 노이즈가 된다.

### 2. 3차 곡선 보간

높은 정확도와 자연스러운 데이터 보간을 위해서 3차 곡선 보간을 사용했습니다. 60Hz에서 획득된 데이터를 3차 곡선 보간을 이용해서 240Hz로 획득된 데이터와 자연스럽게 이어질 수 있게 240Hz로 데이터를 보간했습니다. 이는 궤적의 2차 도함수까지 연속성을 보장할 수 있었습니다. 

## 결론

iPad와 Apple Pencil은 그 자체로 완벽한 측정 장비는 아니다. 하지만 본 연구에서처럼 (1) 240Hz의 고속 샘플링을 확보하고, (2) 내재적 한계(60Hz, zOffset)를 명확히 인식하며, (3) **'3차 곡선 보간'**과 **'물리 기반 보정'**을 통해 데이터를 '적극적으로 정제'한다면, Trjaeco 프로젝트의 Ground Truth로서 충분한 학술적 신뢰도를 제공한다고 최종 판단했다.

- Apple Developer Documentation. *Optimizing ProMotion Refresh Rates for iPhone 13 Pro and iPad Pro.*
- Apple Developer Documentation. *Displays - iOS Device Compatibility Reference (Archived).*
- Apple Developer Documentation (UIKit). `coalescedTouches(for:)`.
- Apple Developer Documentation (UIKit). `predictedTouches(for:)`.
- Apple Developer Documentation (UIKit). *Adopting Hover Support for Apple Pencil.*
- Apple Developer Documentation (UIKit). `UIHoverGestureRecognizer` 및 *zOffset*, *rollAngle* 속성.
- Apple Developer Documentation (UIKit). `touchesEstimatedPropertiesUpdated(_:)` 및 `estimatedProperties`.
- Dui, L., Lomurno, E., Lunardini, F., Termine, C., Campi, A., Matteucci, M., & Ferrante, S. (2022). "Identification and characterization of learning weakness from drawing analysis at the pre-literacy stage." *Scientific Reports.* (인용: /의 관련 연구)
- D'Uscio, A., et al. (2024). "High-Accuracy Biometric Authentication Using Dynamic Handwriting Features from Apple Pencil." *Proceedings of the NDSS Symposium 2024.* (보고서 내  기반 분석)
- Li, A., Li, J., Chai, J., Wu, W., & Chaudhary, S. (2024). "A Study on Cognitive Impairment Assessment Using iPad and Apple Pencil." *JMIR mHealth and uHealth.* (보고서 내  기반 분석)
- Asselbergs, E., et al. (2024). "Using an iPad and Apple Pencil for Quantitative Assessment of Handwriting Process in Children." *Frontiers in Pediatrics (PMC).* (보고서 내  기반 분석)
- The Seven Pens (Documentation). *Parallax vs pen tracking accuracy.*
- Microsoft Corporation (2019). *Touch Accuracy (Windows Hardware Component Guidelines).*
- Vicon Motion Systems. *Vicon Metrology Solutions / Accuracy & Precision.*
- OptiTrack. *Motive: Specs / Accuracy.*