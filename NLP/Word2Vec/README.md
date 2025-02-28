# Efficient Estimation of Word Representations in Vector Space

<br>

## Abstract

<br>

- 본 논문에서는 매우 큰 데이터셋에서 연속적인 단어 벡터(continuous vector representations of words)를 계산하기 위한 2가지 모델 아키텍처를 소개함
- 이러한 표현의 품질은 단어 유사도로 측정하며, 그 결과 훨씬 낮은 계산 비용으로도 정확도가 매우 향샹됨을 확인함
- 또한 단어 벡터가 syntactic & semantic 단어 유사도 측정을 위한 test set에서도 최첨단 성능을 보임

<br>

## Introduction

<br>

- 이전 NLP 시스템에선 단어를 가장 작은 원자 단위로 취급하여 vocabulary 내 인덱스로 표현함 -> 단어간 유사도 개념이 존재하지 않음
- 이러한 방법은 대규모 데이터셋에서 단순한 모델을 학습하는 것이 더 적은 데이터에서 훈련한 복잡한 모델보다 성능이 더 높다는 점에서 장점을 가짐
- but 이러한 단순한 기법은 다양한 task에서 제한됨

<br>

- 본 논문의 목표는 수십억 개의 단어와 수백만 개의 vocabulary를 포함하는 대규모 데이터셋에서 고품질의 단어 벡터를 학습할 수 있는 기법을 소개하는 것임
- 단어 간 유사성은 syntactic 규칙성을 넘어서 단어 벡터에 대한 대수적 연산(+, -, ...)을 통해서도 의미적 의미를 반영할 수 있음
   - $vector("King") - vector("Man") + vector("Woman") ≈ vector("Queen")$
