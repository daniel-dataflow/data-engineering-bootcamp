import React, { useState } from "react";

export default function B_StateObjectUse2() {
  const [members, setMembers] = useState([]);
  const [member, setMember] = useState({
    id: "",
    password: "",
    name: "",
    email: "",
  });
  const addMember = () => {
    setMembers((members) => {
      return [...members, member];
    });
    setMember({
      id: "",
      password: "",
      name: "",
      email: "",
    });
  };
  const insertMember = (e) => {
    const { name, value } = e.target;
    setMember({ ...member, [name]: value });
  };
  return (
    <div>
      <h3>state이용해서 데이터 관리하기</h3>
      <h4>회원정보 출력하기</h4>
      {members.length > 0 ? (
        <table>
          <thead>
            <tr>
              {Object.keys(members[0]).map((k) => (
                <th>{k}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {members.map((member) => (
              <tr>
                {Object.values(member).map((m) => (
                  <td>{m}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <h3>조회된 회원이 없습니다</h3>
      )}
      <div>
        <input
          type="text"
          name="id"
          placeholder="아이디입력"
          onChange={insertMember}
          value={member.id}
        />
        <br />
        <input
          type="password"
          name="password"
          placeholder="패스워드"
          onChange={insertMember}
          value={member.password}
        />
        <br />
        <input
          type="text"
          name="name"
          placeholder="이름"
          onChange={insertMember}
          value={member.name}
        />
        <br />
        <input
          type="email"
          name="email"
          placeholder="이메일"
          onChange={insertMember}
          value={member.email}
        />
        <br />
        <button onClick={addMember}>회원추가</button>
      </div>
    </div>
  );
}
