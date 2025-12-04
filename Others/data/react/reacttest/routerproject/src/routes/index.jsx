import HomeComponent from "../components/HomeComponent";
import LinkTestComponent from "../components/LinkTestComponent";
import AboutPage from "../components/AboutPage";
import NavigateTestComponent from "../components/NavigateTestComponent";
import UserDetailPage from "../components/UserDetailPage";
import UserListComponent from "../components/UserListComponent";
import UseQueryStringComponent from "../components/UseQueryStringComponent";
import UsersContainer from "../components/UsersContainer";

export const routes = [
  { path: "/", element: <HomeComponent /> },
  { path: "/linktest", element: <LinkTestComponent /> },
  { path: "/about", element: <AboutPage /> },
  { path: "/navigatetest", element: <NavigateTestComponent /> },
  { path: "/user/:id", element: <UserDetailPage /> },
  { path: "/userquery", element: <UseQueryStringComponent /> },
  //중첩라우터는 children속성을 이용
  {
    path: "/users",
    element: <UsersContainer />,
    children: [
      { index: true, element: <UserListComponent /> },
      { path: ":id", element: <UserDetailPage /> },
    ],
  },
];
