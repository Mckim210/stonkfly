# 거래 단위 보상: `--reinforcement roundtrip`, `roundtrip-replay`

모의거래 전용 실험 옵션이다. 기본값 `--reinforcement portfolio`는 원래 STONKFLY 그대로다([모델](model.md#what-changes-with-profit-and-loss)). `--live`, `--policy random`, `--shuffle-reinforcement`와는 함께 쓸 수 없다.

```sh
python -m stonkfly run --replay tapes/T.jsonl --reinforcement roundtrip --out runs/rt
python -m stonkfly run --replay tapes/T.jsonl --reinforcement roundtrip-replay --out runs/rr
```

## 세 방식

| 옵션 | 펄스가 들어가는 때 | 무엇의 부호인가 |
| --- | --- | --- |
| `portfolio` (기본) | 매 관측 시작 | 직전 관측 이후 포트폴리오 평가액(매수호가 기준) 변화, ±0.01 USDC 데드밴드 |
| `roundtrip` | 매도가 체결된 **다음 관측** 시작 | 그 매도분의 FIFO 실현 손익(매수·매도 수수료 포함). 0보다 크면 보상, 작으면 혐오, 정확히 0이면 없음 |
| `roundtrip-replay` | `roundtrip`과 같음, 그리고 매도 체결 직후 추가 단계 1번 | 같은 실현 손익의 부호 |

펄스 세기와 길이는 세 방식 모두 같다: 200 ms, 20 mV 상당 전류, 보상은 PAM11 15개, 혐오는 PPL101 2개. 매도 체결이 없는 관측은 신호가 없다(`none`). 마지막 관측에서 체결된 매도는 다음 관측이 없어 전달되지 않고 `final.json`의 `undelivered_final_sale`에 남는다.

## 실현 손익 계산

- 매수 체결 하나가 lot 하나다. 매도 체결은 가장 오래된 lot부터 소진한다(FIFO).
- 소진한 수량 x에 대해 원가 = lot 매수대금 × x / lot 수량, 매수 수수료도 같은 비율로 나눈다.
- 수수료 전 실현 손익 = 매도대금 − 원가. 수수료 후 = 매도대금 − 매도 수수료 − 원가 − 비례 매수 수수료. 신호는 수수료 후 값의 부호다.
- 모두 `Decimal`이다. 원장의 체결 기록(`orders` 테이블)에서 계산하고, lot 상태는 관측과 같은 트랜잭션으로 커밋한다. 중단 후 같은 명령으로 재개해도 결과가 같다(테스트로 확인). 미정 주문이 남아 있으면 계산하지 않고 멈춘다.

## `roundtrip-replay`의 추가 단계

- **언제:** 매도가 체결된 직후, 다음 관측의 정규 단계 전. 신경 시간 500 ms(관측과 같은 길이), 처음 200 ms에 펄스.
- **무엇을 보여 주나:** 그 매도분의 첫 번째 FIFO lot을 산 관측에서 초파리가 본 RGB 차트. 매수 주문 의도(주문 전에 원장에 기록된 관측의 가격 이력과 호가)로 다시 그리고, 그때 기록한 입력 해시(`input_sha256`)와 같은지 확인한다. 다르면 실행을 멈춘다.
- **제안을 읽지 않는다:** `FlyController.replay`는 `observe`와 같은 신경 적분을 하지만 DNp20/DNpe017 해독을 하지 않는다. 제안, 주문, 관측 수에 들어가지 않는다. 전체 커넥톰 테스트에서 같은 상태로부터 `replay`와 `observe`의 스파이크와 기억이 같음을 확인했다.
- **왜:** 기억 규칙의 KC·DAN 흔적은 신경 시간 1초다. 관측 하나가 500 ms라서, 매도 결과가 나올 때쯤 매수 결정 때의 KC 활동은 흔적에 거의 남지 않는다. 매수 때 화면을 다시 보여 주면 비슷한 KC 활동이 펄스와 겹칠 수 있다.

## 원본 STONKFLY와 다른 점

1. **보상의 정의.** 원본은 매 관측 포트폴리오 평가손익(미실현 포함)이다. 여기서는 매도 1건의 실현 손익만 쓴다. 보유 중 가격 변화와 매수 수수료는 그 자체로는 신호가 되지 않고, 매도 때 손익에 들어간다.
2. **데드밴드 없음.** 원본은 ±0.01 USDC 안의 변화를 무시한다. 여기서는 0보다 크거나 작으면 신호다.
3. **빈도.** 원본은 거의 매 관측 신호가 있을 수 있다. 여기서는 매도 체결 다음 관측에만 있다.
4. **(replay) 관측 사이의 추가 신경 시간.** 매도 1건마다 500 ms가 더해지므로 `brain_ms`가 관측 수 × 500 ms보다 크다. 추가 단계는 과거 화면이지 새 시장 관측이 아니다. 원본에 없는 인위적 재생 자극이며, 초파리의 생물학적 재생(replay) 현상을 모델링한다는 주장이 아니다.
5. **(replay) 짝짓기가 깨끗하지 않다.** 펄스는 추가 단계 처음 200 ms에 들어가는데, KC는 화면이 바뀐 뒤 늦게 반응하고 1초 흔적에는 직전 관측(매도 관측) 화면의 KC 활동이 남아 있다. 그래서 신호는 매수 화면 KC뿐 아니라 매도 관측 화면 KC와도 짝지어진다. 다음 정규 관측의 펄스는 replay 단계의 매수 화면 KC 흔적과도 짝지어진다.
6. **(replay) 같은 화면이 같은 KC를 깨운다는 보장이 없다.** 네트워크 상태, KC 적응, 바뀐 시냅스가 다르다. 예: 합성 테이프로 한 배관 점검(관측 40개) 한 번에서 새 뇌의 관측 1–2는 KC 4개만 발화했고 관측 3부터는 약 1,400–1,600개가 발화했다. 관측 1에 산 lot을 관측 37에 다시 보여 주자 KC 1,476개가 발화해 Jaccard가 0이었다. 그래서 겹침(Jaccard)을 기록한다.
7. **기록이 늘어난다.** 이 옵션에서만 `events.jsonl` 각 행에 `roundtrip`, `final.json`에 `reinforcement`와 `roundtrip` 요약이 생긴다. 기본 실행의 기록과 동작은 바뀌지 않는다(같은 fixture 입력에서 원본 코드와 스파이크 해시, 제안, 기억, 신호, 체결, `final.json`이 같음을 확인).
8. **모의거래 전용.**

## 기록

`events.jsonl`의 `roundtrip` (관측마다):

| 필드 | 뜻 |
| --- | --- |
| `sale` | 직전 관측에 체결된 매도의 FIFO 계산(매칭된 lot, 매수 관측 번호, 원가, 수수료) 또는 `null` |
| `realized_pnl_before_fees`, `realized_pnl_after_fees` | 실현 손익(USDC 문자열) 또는 `null` |
| `replay`, `buy_observation` | replay 단계의 신경 기록(스파이크, 기억, `kc_active`, 입력 해시)과 첫 FIFO 매수 관측 번호, 또는 `null` |
| `memory_before` | 이 관측 처리 전(replay 전)의 기억 상태 |
| `kc_active` | 정규 단계에서 한 번 이상 발화한 KC의 그래프 인덱스 |
| `edges_changed_in_step` | 정규 단계 동안 가중치가 바뀐 가소성 시냅스 수 (replay는 `replay` 안에 따로) |
| `portfolio_signal` | 같은 관측에서 기본 방식이었다면 줬을 신호(참고용, 전달 안 함) |

`reinforcement_earned`와 `reinforcement_delivered`는 이 옵션에서 거래 단위 신호다. `final.json`의 `roundtrip`: 거래 단위 보상/혐오 신호 수, replay 포함 펄스 수, 실현 손익 부호별 매도 수, 펄스마다 `changed_edges`·`mean_efficacy` 전후, replay KC 집합과 원래 매수 시점 KC 집합의 Jaccard.

## 해석 한계

보상은 여전히 엔지니어링된 입력이다. 고통, 쾌락, 의식은 모델링하지 않는다. 해독기는 고정이고 보상은 KC→MBON07/11 시냅스만 바꾼다. 바뀐 시냅스가 매수·매도 제안을 더 나은 쪽으로 바꾸는 경로는 설계되어 있지 않다. 이 옵션은 수익을 내는 학습을 검증하지 않는다.
