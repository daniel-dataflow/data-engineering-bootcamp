import React, { useState } from "react";
import { useDispatch } from "react-redux";
import { addTodo } from "../../features/todos/todoSlice";
export default function TodoInput() {
  const [text, setText] = useState("");
  const dispatch = useDispatch();

  const inputText = (e) => {
    setText(e.target.value);
  };
  const saveTodo = (e) => {
    dispatch(addTodo(text));
    setText("");
  };
  return (
    <div>
      <input type="text" name="text" value={text} onChange={inputText} />
      <button onClick={saveTodo}>저장</button>
    </div>
  );
}
