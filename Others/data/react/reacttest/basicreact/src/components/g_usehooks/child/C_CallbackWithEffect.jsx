import React, { useEffect, useState, useCallback } from "react";

export default function C_CallbackWithEffect() {
  const [users, setUsers] = useState([]);
  //   const fetchUsers = async () => {
  //     console.log("유저 목록 요청중....");
  //     const response = await fetch("https://jsonplaceholder.typicode.com/users");
  //     const data = await response.json();
  //     setUsers(data);
  //   };
  // 요청을 보내는 함수에 useCallback()을 적용 -> 무한루프가 되지 않음.
  const fetchUsers = useCallback(async () => {
    console.log("유저 목록 요청중....");
    const response = await fetch("https://jsonplaceholder.typicode.com/users");
    const data = await response.json();
    setUsers(data);
  }, []);
  // 페이지가 로딩하면 바로 요청을 보내기
  // 무한으로 요청을 보냄
  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);
  return (
    <div>
      <h3>useEffect함수에서 다른 함수를 호출해서 사용할때 무한루프방지하기</h3>
      <ul>
        {users.map((u) => {
          return <li key={u.id}>{u.name}</li>;
        })}
      </ul>
    </div>
  );
}
