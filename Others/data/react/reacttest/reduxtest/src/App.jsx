import CounterComponent from "./components/CounterComponent";
import TodoContainer from "./components/TodoContainer";
import PostContainer from "./components/PostContainer";
function App() {
  return (
    <>
      <h2>RTK를 이용한 리덕스 적용하기</h2>
      <h3>counter기능 구현하기</h3>
      <CounterComponent />
      <h3>todolist구현하기</h3>
      <TodoContainer />
      <h3>비동기요청을 처리하기</h3>
      <PostContainer />
    </>
  );
}

export default App;
