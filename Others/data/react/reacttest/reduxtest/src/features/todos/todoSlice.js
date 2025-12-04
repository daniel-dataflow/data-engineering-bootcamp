import { createSlice } from "@reduxjs/toolkit";

export const TodoFilter = {
  ALL: "ALL",
  DONE: "DONE",
  ACTIVE: "ACTIVE",
};
Object.freeze(TodoFilter);

const initialData = {
  items: [], //todo내용이 들어 가는 속성
  filter: TodoFilter.ALL, //데이터 필터 내용이 들어가는 속성 ALL, DONE, ACTIVE
};
const todoGenerator = (function* (title) {
  let num = 0;
  while (true) {
    yield `${title}_${++num}`;
  }
})("todo");

const todoSlice = createSlice({
  name: "todoSlice",
  initialState: initialData,
  reducers: {
    //리듀서를 객체로 지정하여 추가 옵션설정하기
    addTodo: {
      reducer(state, action) {
        state.items.push(action.payload);
      }, //text에는 dispatch()시 action의 매개변수로 전달된 데이터가 저장
      prepare(text) {
        //미리 payload구조를 설정할 수 있음.
        //반환되는 객체를 payload로 설정
        return {
          payload: { id: todoGenerator.next().value, text: text, done: false },
        };
      },
    },
    toggleTodo(state, action) {
      const todo = state.items.find((todo) => todo.id == action.payload);
      if (todo) {
        todo.done = !todo.done;
      }
    },
    removeTodo(state, action) {
      state.items = state.items.filter((todo) => todo.id != action.payload);
    },
    setFilter(state, action) {
      state.filter = action.payload;
    },
  },
});

export const { addTodo, toggleTodo, removeTodo, setFilter } = todoSlice.actions;

export default todoSlice.reducer;
