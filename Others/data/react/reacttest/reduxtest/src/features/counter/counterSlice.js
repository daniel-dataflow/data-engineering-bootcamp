//reducer구성하기
import { createSlice } from "@reduxjs/toolkit";
//1. 관리할 데이터 초기상태 설정
const initialState = {
  value: 0,
};

//2. slice생성 -> dispatch() 요청을 받아서 store에 저장된 값을 수정하는 객체
const counterSlice = createSlice({
  name: "counter", //슬라이트 이름설정
  initialState, // 관리할 데이터 초기값 설정 -> 구조설정
  reducers: {
    // dispatch()함수 호출해서 데이터를 조작하는 함수정의
    // 함수명이 action이 됨
    increment(state) {
      state.value += 1;
    },
    decrement(state) {
      state.value -= 1;
    },
    incrementByAmount(state, action) {
      state.value += action.payload;
    },
    setValue(state, action) {
      state.value = action.payload;
    },
  },
});

//action,reducer export하기
export const { increment, decrement, incrementByAmount, setValue } =
  counterSlice.actions;

// 기본으로 reducer를 반환함.
export default counterSlice.reducer;
