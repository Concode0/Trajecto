# Jacobian 유도 및 쿼터니언 기반 강체 변환 유도

## **1. 이론적 기반의 필요성**

Trajecto 프로젝트의 핵심 엔진은 EKF(확장 칼만 필터)입니다. 하지만 EKF를 단순한 라이브러리로 사용하는 것은 '블랙박스'에 의존하는 것과 같습니다. $Q$ 와  $R$ 행렬을 튜닝하고, '물리적 역설'과 같은 복잡한 문제를 해결하기 위해서는, 저는 이 엔진의 가장 깊은 곳, 즉 '상태 천이 모델($f$)'과 '측정 모델($h$)'의 수학적 유도 과정을 이해하고 설계하기로 했습니다.

## **2. 쿼터니언(Quaternion)과 강체 변환(Rigid Body Transformation)의 당위성**

### 왜 쿼터니언 일까?

3D 공간의 자세를 표현할 때 오일러 각(Euler Angles)을 사용하면, 특정 자세에서 자유도를 잃는 '짐벌 락(Gimbal Lock)'이라는 치명적인 문제에 부딪힙니다. 쿼터니언은 이 문제를 원천적으로 해결하며, 4차원 벡터($q = [q_w, q_x, q_y, q_z]$)를 이용해 3D 회전을 효율적이고 강건하게 표현합니다.

### 왜 강체 변환인가?

IMU 센서는 '펜의 몸체(Body Frame)'에서 가속도를 측정하지만, 우리가 추적해야 하는 것은 '세상(World Frame)' 기준의 궤적입니다. 또한, EKF가 추정하는 'IMU 센서'의 위치와 우리가 최종적으로 원하는 '펜 팁(Pen Tip)'의 위치는 물리적으로 떨어져 있습니다. '강체 변환'은 EKF가 추정한 자세(쿼터니언)를 이용해, 센서 좌표계의 측정값($v^b$)을 월드 좌표계의 값($v^w$)으로 변환($v^w$ =  $q \otimes v^b \otimes q^*$)하고, '펜 팁'의 가속도까지 계산하는 모든 물리적 변환의 핵심입니다.

## **3. EKF와 Jacobian의 필요성**

이 쿼터니언 기반의 '강체 변환'과 '자세 예측' 과정은 '비선형(Non-linear)'입니다. EKF의 핵심은 이 비선형 상태 천이 모델($f$)과 측정 모델($h$)을 1차 테일러 급수 전개로 '선형화(Linearize)'하는 것입니다.
이 선형화에 필요한 것이 바로 '야코비안 행렬(Jacobian Matrix)', 즉 각 변수에 대한 편미분 행렬입니다.

$$
F_k = \frac{\partial f}{\partial x} \bigg|_{x_{k-1}, u_k} \quad \text{and} \quad H_k = \frac{\partial h}{\partial x} \bigg|_{x_k}
$$

## 4. 비선형 상태 천이 모델 $f(x, u)$ 정의

EKF의 핵심은 비선형 모델 $x_k = f(x_{k-1}, u_k)$을 선형화하는 것입니다. 저는 16차원 상태 벡터 $x = [position,  velocity,  quaternion,  b_{accel},  b_{gyro}]$과 IMU 입력 $u = [u_{accel}, u_{gyro}]$에 대해, Trajecto의 '물리 방정식'을 다음과 같이 정의했습니다.

### 1. 위치(p) 예측

$$
\\ p_k = p_{k-1} + v_{k-1} \Delta t + \frac{1}{2} a_{\text{world}} \Delta t^2
$$

### 2. 속도(v) 예측

$$
\\ a_{\text{world}} = C(q_{k-1})(u_a - b_{a,k-1}) - g_{\text{world}} \\
v_k = v_{k-1} + a_{\text{world}} \Delta t
$$

$C(q)$는 쿼터니언 q로부터 유도된 회전 행렬이다.

### 3. 자세(q) 예측

$$
\\ \omega_{\text{corrected}} = u_g - b_{g,k-1} \\
q_k = q_{k-1} \otimes \Delta q(\omega_{\text{corrected}}, \Delta t)
$$

### 4. 바이어스(b) 예측

$$
\\ b\_{a,k} = b\_{a,k-1} + w\_{ba} \\
b\_{g,k} = b\_{g,k-1} + w\_{bg}
$$

## 5. 핵심 유도 과정 : 16x16 야코비안 $F_k$

저는 시스템의 비선형성을 결정하는 핵심 항들을 집중적으로 유도했습니다. $I$는 단위행렬, $0$은 영행렬입니다.

$$
F_k =
\begin{bmatrix}
\frac{\partial p}{\partial p} & \frac{\partial p}{\partial v} & \frac{\partial p}{\partial q} & \frac{\partial p}{\partial b_a} & \frac{\partial p}{\partial b_g} \\
\frac{\partial v}{\partial p} & \frac{\partial v}{\partial v} & \frac{\partial v}{\partial q} & \frac{\partial v}{\partial b_a} & \frac{\partial v}{\partial b_g} \\
\frac{\partial q}{\partial p} & \frac{\partial q}{\partial v} & \frac{\partial q}{\partial q} & \frac{\partial q}{\partial b_a} & \frac{\partial q}{\partial b_g} \\
\frac{\partial b_a}{\partial p} & \frac{\partial b_a}{\partial v} & \frac{\partial b_a}{\partial q} & \frac{\partial b_a}{\partial b_a} & \frac{\partial b_a}{\partial b_g} \\
\frac{\partial b_g}{\partial p} & \frac{\partial b_g}{\partial v} & \frac{\partial b_g}{\partial q} & \frac{\partial b_g}{\partial b_a} & \frac{\partial b_g}{\partial b_g}
\end{bmatrix}

$$

**대각선 및 단순항**

$$
\frac{\partial p}{\partial p} = I, \frac{\partial v}{\partial v} = I, \frac{\partial b_a}{\partial b_a} = I, \frac{\partial b_g}{\partial b_g} = I (Random Walk)
* \frac{\partial p}{\partial v} = I \Delta t
$$

**위치-속도 관계 ( 서로 독립적인 항 )**

$$
 \frac{\partial v}{\partial p} = 0, \frac{\partial q}{\partial p} = 0 
$$

**핵심 비선형 항 유도**

$\frac{\partial v_k}{\partial q_{k-1}}$이 항이 EKF의 핵심입니다. 이는 속도 예측식을 쿼터니언에 대해 편미분하는 과정입니다. 이는 쿼터니언 미적분을 사용해 전개해야하는 복잡한 항입니다. 

**가속도 바이어스가 속도에 미치는 영향**

$$
\frac{\partial v_k}{\partial b_{a, k-1}} = -C(q_{k-1}) \Delta t
$$

**자세가 속도에 미치는 영향**

$$
\frac{\partial v_k}{\partial q_{k-1}} = \frac{\partial (C(q)a_{\text{body}})}{\partial q} \Delta t
$$

**자이로 바이어스가 자세에 미치는 영향**

$$
\frac{\partial q_k}{\partial b_{g, k-1}} = -\frac{\partial q_k}{\partial \omega_{\text{corrected}}}
$$

## 6**. 측정 야코비안 $H_k$**

마찬가지로, 측정 모델 $z_k = h(x_k)$의 야코비안 $H_k = \frac{\partial h}{\partial x}$도 직접 유도했습니다.
• **중력 측정:** 측정값 $z_{\text{accel}}$ (가속도계가 측정한 중력 방향)은 $h(x_k) = C(q_k)^T g_{\text{world}}$ 로 모델링됩니다. 이를 $q$에 대해 편미분하여 $H_k$를 구했습니다.
• **ZUPT 측정:** ZUPT가 감지되면, 측정값 $z_{\text{ZUPT}} = [0, 0, 0]$ (속도=0)을 사용합니다. 측정 모델은 $h(x_k) = v_k$가 됩니다. 이는 16차원 상태 벡터 $x$에 대해 편미분하면 $H_{\text{ZUPT}} = \begin{bmatrix} 0 & I_{3 \times 3} & 0 & 0 & 0 \end{bmatrix}$ (속도 $v$에 대한 항만 $I$가 됨) 라는 매우 강력하고 간단한 선형 측정 모델이 됩니다.