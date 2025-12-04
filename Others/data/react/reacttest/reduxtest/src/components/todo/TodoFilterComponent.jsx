import React from "react";
import { useDispatch } from "react-redux";
import { TodoFilter } from "../../features/todos/todoSlice";
import { setFilter } from "../../features/todos/todoSlice";

export default function TodoFilterComponent() {
  const dispatch = useDispatch();
  const filterSave = (id) => {
    dispatch(setFilter(id));
  };
  return (
    <div>
      <select onChange={(e) => filterSave(e.target.value)}>
        {Object.values(TodoFilter).map((filter) => (
          <option key={filter} value={filter}>
            {filter}
          </option>
        ))}
      </select>
    </div>
  );
}
