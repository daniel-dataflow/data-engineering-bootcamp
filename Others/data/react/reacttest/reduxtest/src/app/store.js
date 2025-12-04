import { configureStore } from "@reduxjs/toolkit";
import counterReducer from "../features/counter/counterSlice";
import todoReducer from "../features/todos/todoSlice";
import postsReducer from "../features/posts/postSlice";
//store설정하기 -> 저장소만들기
export const store = configureStore({
  //생성한 Reducer를 설정해서 저장공간을 이용
  reducer: {
    counter: counterReducer,
    todos: todoReducer,
    posts: postsReducer,
  },
});
