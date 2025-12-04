import React, { useReducer } from "react";
import { formDataReducer } from "../reducers/reducer";
export default function D_ReducerFormComponent() {
  const [state, dispatch] = useReducer(formDataReducer, {});
  const changeFormDataHandler = (e) => {
    // 구조분해할당으로 데이터 받기
    const { name, type, value } = e.target;
    dispatch({
      type: "ADD",
      field: name,
      payload: type == "checkbox" ? value == "on" : value,
    });
  };
  return (
    <div>
      <h4>회원가입</h4>
      <form style={{ display: "flex", flexDirection: "column" }}>
        <input
          type="text"
          name="userId"
          value={state.userId}
          onChange={changeFormDataHandler}
        />
        <input
          type="password"
          name="userPw"
          value={state.userPw}
          onChange={changeFormDataHandler}
        />
        <input
          type="text"
          name="userName"
          value={state.userName}
          onChange={changeFormDataHandler}
        />
        <label>
          <input
            type="checkbox"
            name="agreed"
            checked={state.agreed}
            onChange={changeFormDataHandler}
          />
          개인정보동의
        </label>
        <button
          type="button"
          onClick={() => {
            dispatch({ type: "RESET" });
          }}
        >
          저장취소
        </button>
      </form>
      <div>
        {state.userId && (
          <>
            <h4>등록된 정보</h4>
            {Object.entries(state).map((k, v) => {
              return (
                <p key={`${k}_${v}`}>
                  {k[0]} : {k[1]}
                </p>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}
