import React from "react";
import { Outlet } from "react-router-dom";
import HeaderComponent from "./common/HeaderComponent";

export default function UsersContainer() {
  return (
    <div>
      <HeaderComponent />
      <h2>중첩라우터이용하기</h2>
      <Outlet />
    </div>
  );
}
