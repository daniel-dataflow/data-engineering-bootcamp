import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { users } from "../resources/commondata";
import HeaderComponent from "./common/HeaderComponent";
export default function UseQueryStringComponent() {
  //queryString 값을 가져와 처리하려면
  // useSearchParams() Hook을 이용
  const [search, setSearch] = useSearchParams();
  const [user, setUser] = useState(null);
  useEffect(() => {
    // const param = {};
    // search.forEach((value, name) => (param[name] = value));
    // 배열을 객체로 변경해주는 함수 -> index가 Key, value가 value가 됨.
    const param = Object.fromEntries(search);
    let searchUsers = [];
    Object.entries(param).forEach((value) => {
      searchUsers = searchUsers.concat(
        users.filter((u) => {
          switch (value[0]) {
            case "age":
              return u[value[0]] >= Number(value[1]);
            case "name":
              return u[value[0]].includes(value[1]);
            default:
              return u[value[0]] == value[1];
          }
        })
      );
    });
    setUser(searchUsers);
  }, [search]);
  return (
    <div>
      <HeaderComponent />
      <h3>querystring값을 이용해서 데이터 출력하기</h3>
      {user && (
        <table>
          <thead>
            <tr>
              {Object.keys(user[0]).map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {user.map((user) => (
              <tr key={user.id}>
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
      )}
    </div>
  );
}
