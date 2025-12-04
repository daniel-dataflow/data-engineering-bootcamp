import React from "react";
import TodoList from "./todo/TodoList";
import TodoInput from "./todo/TodoInput";
import TodoFilterComponent from "./todo/TodoFilterComponent";

export default function TodoContainer() {
  return (
    <div>
      <TodoFilterComponent />
      <TodoList />
      <TodoInput />
    </div>
  );
}
