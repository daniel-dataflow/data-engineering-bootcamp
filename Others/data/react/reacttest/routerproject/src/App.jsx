import { Routes, Route, useRoutes } from "react-router-dom";
// Routes, Route컴포넌트를 이용해서 url과 컴포넌트를 연결
import HomeComponent from "./components/HomeComponent";
import AboutPage from "./components/AboutPage";
import UserDetailPage from "./components/UserDetailPage";
import UserListComponent from "./components/UserListComponent";
import LinkTestComponent from "./components/LinkTestComponent";
import NavigateTestComponent from "./components/NavigateTestComponent";
import UseQueryStringComponent from "./components/UseQueryStringComponent";
import UsersContainer from "./components/UsersContainer";
import { routes } from "./routes";

function App() {
  //라우터를 객체화해서 관리하기
  //useRoutes()hook을 이용해서 태그로 만들지 않고 객체로 모듈화해서 관리할 수 있음.
  const element = useRoutes(routes);
  return element;
  // return (
  //   <>
  //     <Routes>
  //       {/* 기본정보 확인하기 */}
  //       <Route path="/" element={<HomeComponent />} />
  //       <Route path="/about" element={<AboutPage />} />
  //       <Route path="/linktest" element={<LinkTestComponent />} />
  //       <Route path="/navigatetest" element={<NavigateTestComponent />} />

  //       {/* url주소에 동적경로를 설정할때 :key로 설정함 */}
  //       <Route path="/user/:id" element={<UserDetailPage />} />
  //       <Route path="/userquery" element={<UseQueryStringComponent />} />
  //       {/* 중첩라우트 설정하기 */}
  //       <Route path="/users" element={<UsersContainer />}>
  //         {/* 중첩라우트로 연결됐을때 기본으로 연결되는 페이지에 index속성을 설정 */}
  //         <Route index element={<UserListComponent />} />
  //         <Route path=":id" element={<UserDetailPage />} />
  //       </Route>
  //     </Routes>
  //   </>
  // );
}

export default App;
