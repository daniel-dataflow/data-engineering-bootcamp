import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
//actions 정보를 import하기
import {
  increment,
  decrement,
  incrementByAmount,
  setValue,
} from "../features/counter/counterSlice";
export default function CounterComponent() {
  //useSelector()훅을 이용해서 store에 저장된 counter값 가져오기
  //state객체는:??
  const count = useSelector((state) => state.counter.value);
  //useDispatch()를 이용해서 store에 접근해서 수정할 수 있는 dispatch()함수 가져오기
  const dispatch = useDispatch();

  const [amount, setAmount] = useState(0);
  return (
    <div>
      <h4>현재 카운트 : {count}</h4>
      <div>
        <h4>store값 증가/감소하기</h4>
        <p>increment(), decrement()를 호출해서 변경</p>
        <button
          onClick={() => {
            dispatch(increment());
          }}
        >
          증가
        </button>
        <button
          onClick={() => {
            dispatch(decrement());
          }}
        >
          감소
        </button>
        <div>
          <h3>필요한 만큼 더하기</h3>
          <input
            type="text"
            onChange={(e) => {
              setAmount(parseInt(e.target.value));
            }}
          />
          <button
            onClick={() => {
              dispatch(incrementByAmount(amount));
            }}
          >
            추가하기
          </button>
        </div>
      </div>
    </div>
  );
}
