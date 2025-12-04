import React from "react";
import { useSelector, useDispatch } from "react-redux";
import { TodoFilter } from "../../features/todos/todoSlice";
import { toggleTodo, removeTodo } from "../../features/todos/todoSlice";
export default function TodoList() {
  //TodoList를 스토어에서 가져오기
  const items = useSelector((state) => state.todos.items);
  //Filter를 위해 filter내용가져오기
  const todoFilter = useSelector((state) => state.todos.filter);

  const dispatch = useDispatch();
  const removeHandler = (id) => {
    dispatch(removeTodo(id));
  };
  const toggleHandler = (id) => {
    dispatch(toggleTodo(id));
  };
  //필터링된 데이터만 출력하하게 items필터링
  const filterItems = items.filter((todo) => {
    switch (todoFilter) {
      case TodoFilter.ACTIVE:
        return !todo.done;
      case TodoFilter.DONE:
        return todo.done;
      default:
        return true;
    }
  });
  return (
    <div>
      <h3>todoList</h3>
      <ul>
        {filterItems.map((todo) => (
          <li
            key={todo.id}
            onClick={(e) => {
              toggleHandler(todo.id);
            }}
          >
            <span style={{ textDecoration: todo.done ? "line-through" : "" }}>
              {todo.text}
            </span>{" "}
            <button
              onClick={(e) => {
                e.stopPropagation;
                removeHandler(todo.id);
              }}
            >
              삭제
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
