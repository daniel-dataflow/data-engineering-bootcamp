import React from "react";
import { users } from "../resources/commondata";
import HeaderComponent from "./common/HeaderComponent";
import { useNavigate } from "react-router-dom";
export default function UserListComponent() {
  //회원클릭시 중첩라우터로 이동하는 로직
  const navigate = useNavigate();
  const modveDetail = (id) => (e) => {
    navigate(`/users/${id}`);
  };

  return (
    <div>
      <h3>회원 리스트 출력하기</h3>
      <div>
        <h4>parameter를 이용해서 회원 조회하기</h4>
        <p>
          navigate()함수를 이용해서 특정 이벤트와 연결해서 Url주소를 변경해서
          페이지를 전환할 수 있음
        </p>
      </div>
      <table>
        <thead>
          <tr>
            {Object.keys(users[0]).map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {users.map((user) => (
            // 클릭하면 상세화면으로 이동하는 로직작성
            <tr
              key={user.id}
              onClick={modveDetail(user.id)}
              style={{ cusor: "pointer" }}
            >
              {Object.values(user).map((u) => {
                if (typeof u == "boolean") {
                  return <td key={u}>{u ? "활성화" : "비활성화"}</td>;
                }
                return <td key={u}>{u}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
