import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import HeaderComponent from "./common/HeaderComponent";
import { users } from "../resources/commondata";
export default function UserDetailPage() {
  //동적경로의 정보를 가지고 데이터를 가져와 처리하기
  const param = useParams();
  const [user, setUser] = useState();
  //페이지가 로딩되면 데이터를 가져오기
  //param의 값을 기준으로 가져오기
  useEffect(() => {
    setUser(users.filter((u) => u.id == param.id)[0]);
  }, [param]);
  return (
    <div>
      <HeaderComponent />
      <h3>동적경로로 설정된 데이터 가져오기</h3>
      <p>url주소의 동적경로를 useParams()를 이용해서 가져올 수 있음</p>
      <h3>요청한 사용자에 대한 정보</h3>
      <ul>
        {user &&
          Object.entries(user).map((v) => (
            <li key={v[0]}>{`${v[0]} : ${v[1]}`}</li>
          ))}
      </ul>
    </div>
  );
}
