import React from "react";
import D_ReducerCounterComponent from "./reducer_components/D_ReducerCounterComponent";
import D_ReducerFormComponent from "./reducer_components/D_ReducerFormComponent";
import D_ReducerTodoList from "./reducer_components/D_ReducerTodoList";
export default function D_UseReducerContainer() {
  return (
    <div>
      <h4>useReducer()이용하기</h4>
      <p>데이터의 상태관리를 위해 사용하는 hook</p>
      <p>
        state보다 복잡한 상태를 처리할때 사용, 데이터를 관리하는 로직을 분리할
        수 있어, 복잡한 데이터를 다루는데 유리함
      </p>
      <div>
        useReducer의 데이터 처리구조
        <ol>
          <li>
            1. Reducer함수 생성하기
            <p>
              - 요청하는 내용에 따라 새로운 상태를 반환(불변성 유지!)해주는 함수{" "}
            </p>
            <p>
              - 상태값을 type에 따라 수정, 초기화 등의 데이터를 다루는 작업을
              실행
            </p>
            <p>- state, action의 매개변수를 받는 함수를 선언</p>
            <p>
              - state : 현재 reducer가 관리하고 있는 값/useReducer호출시 전달한
              초기 값이 저장
            </p>
            <p>
              - action : 데이터 조작요청(dispatch)시 전달된 데이터를 저장한 객체
            </p>
            <p>
              action에 저장된 데이터는 구조화해서 관리해야함. 예) type,
              value/payload
            </p>
          </li>
          <li>
            2. useReducer()를 이용해서 reducer함수를 이용
            <p>
              - 매개변수에 선언한 reducer함수(필수)와 초기값(선택)을 전달할 수
              있음
            </p>
            <p>- state,dispatch를 반환해 줌</p>
          </li>
        </ol>
      </div>
      <h2>간단한 카운터리듀서 만들기</h2>
      <D_ReducerCounterComponent></D_ReducerCounterComponent>
      <h2>form에서 입력받은 데이터 관리하기</h2>
      <D_ReducerFormComponent></D_ReducerFormComponent>
      <h2>to_dolist만들기</h2>
      <D_ReducerTodoList></D_ReducerTodoList>
    </div>
  );
}
