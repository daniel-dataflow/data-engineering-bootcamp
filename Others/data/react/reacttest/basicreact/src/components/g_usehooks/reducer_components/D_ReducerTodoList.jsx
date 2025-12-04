import React, { useReducer, useState } from "react";
import { todolistReducer } from "../reducers/reducer";
export default function D_ReducerTodoList() {
  const [state, dispatch] = useReducer(todolistReducer, []);
  const [text, setText] = useState("");
  const textChange = (e) => {
    setText(e.target.value);
  };
  const addTodo = (e) => {
    dispatch({ type: "ADD", text: text });
    setText("");
  };
  const clickHandler = (type, id) => (e) => {
    dispatch({ type: type, id: id });
  };

  return (
    <div>
      <h3>todolist</h3>
      <div>
        <input type="text" onChange={textChange} />
        <button onClick={addTodo}>저장</button>
      </div>
      <div>
        <ul>
          {state.map((todo) => {
            return (
              <li
                key={todo.id}
                style={{
                  textDecoration: todo.done ? "line-through" : "none",
                  cursor: "pointer",
                }}
              >
                <span onClick={clickHandler("TOGGLE", todo.id)}>
                  {todo.text}
                </span>
                <button onClick={clickHandler("REMOVE", todo.id)}>삭제</button>
              </li>
            );
          })}
        </ul>
        <button onClick={clickHandler("CLEAR_DONE")}>완료된 내용 삭제</button>
      </div>
    </div>
  );
}
