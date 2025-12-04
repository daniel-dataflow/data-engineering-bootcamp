import React from "react";
const outerData = "outerData";
let outerLetData = "변경해봐";
export default function B_UseJavascript_Func() {
  //자료형별 변수선언하기
  const name = "유병승";
  let age = 19;
  const height = [180.5, 175.5, 160.5, 155.5];
  const person = {
    name: "유병승",
    age: 19,
    addr: "경기도 시흥시",
  };
  const student = { name: "홍길동", grade: 1, classNum: 2, gender: "남" };
  const students = [
    { name: "홍길동", grade: 1, classNum: 2, gender: "남" },
    { name: "김유신", grade: 2, classNum: 3, gender: "남" },
    { name: "신사임당", grade: 1, classNum: 5, gender: "여" },
    { name: "선덕여왕", grade: 2, classNum: 3, gender: "여" },
  ];

  outerLetData += " 변경할꺼야";

  return (
    <>
      <h2>변수활용하기</h2>
      <h4>이름 : {name}</h4>
      <h4>나이 : {age}</h4>
      {/* 배열은 각 인덱스의 값을 합쳐서 출력함 */}
      <h4>키 : {height}</h4>
      {/* 배열은 출력하지만 객체는 출력할 그냥 출력할 수 없음 */}
      {/* <p>사람 : {person}</p> */}
      {/* <p>학생들 : {student}</p> */}

      <h2>객체, 객체배열 출력</h2>
      <p>
        객체, 배열, 객체배열은 직접 접근하여 각 값에 출력하거나 함수를 이용해서
        출력함
      </p>
      <h3>직접접근하여 출력</h3>
      <h4>배열출력하기</h4>
      <ul>
        <li>{height[0]}</li>
        <li>{height[1]}</li>
        <li>{height[2]}</li>
        <li>{height[3]}</li>
      </ul>
      <h4>객체출력</h4>
      <ul>
        <li>이름 : {person.name}</li>
        <li>나이 : {person.age}</li>
        <li>주소 : {person["addr"]}</li>
      </ul>

      <h4>직접 출력하기 리스트출력</h4>
      <table>
        <thead>
          <tr>
            <th>이름</th>
            <th>학년</th>
            <th>반</th>
            <th>성별</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>{students[0].name}</td>
            <td>{students[0].grade}</td>
            <td>{students[0].classNum}</td>
            <td>{students[0].gender}</td>
          </tr>
          <tr>
            <td>{students[1]["name"]}</td>
            <td>{students[1]["grade"]}</td>
            <td>{students[1]["classNum"]}</td>
            <td>{students[1]["gender"]}</td>
          </tr>
        </tbody>
      </table>

      <h2>함수를 이용해서 다중 데이터(리스트, 배열) 출력하기</h2>
      <p>
        기본 js에서 제공하는 함수이용해서 다중데이터를 출력하기 jsx를
        배열방식으로 {}내부에서 출력하면 된다.
      </p>
      <h4>배열 함수 이용해서 출력하기</h4>
      <p>
        함수를 이용하는 방법은 반환값을 이용한다. 반환되는 값이 단일 jsx이거나
        배열 형식의 jsx구문이면 된다. 일반적으로 배열내용은 map()을 이용해서
        반환되는 내용을 출력하는 구문을 많이 활용함.
      </p>
      <p>반복해서 출력하는 jdx는 key속성을 설정해줘야 함</p>
      <h3>배열 데이터 List로 출력하기 - map이용하기</h3>
      <ul>
        {height.map((v) => (
          <li key={v}>{v}</li>
        ))}
      </ul>
      <h3>filter와 map이용하기</h3>
      <ul>
        {height
          .filter((v) => v > 170)
          .map((v) => (
            <li key={v}>{v}</li>
          ))}
      </ul>

      <h3>일반객체 출력하기</h3>
      <p>Object.values()함수 이용해서 단일 객체 출력하기</p>
      <ul>
        {Object.values(student).map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ul>
      <h3>객체 리스트타입을 테이블 출력하기</h3>
      <p>배열 메소드를 이용해서 테이블 방식으로 출력하기</p>
      <table>
        <thead>
          <tr>
            <th>이름</th>
            <th>학년</th>
            <th>반</th>
            <th>성별</th>
          </tr>
        </thead>
        <tbody>
          {students.map((v) => {
            return (
              <tr key={v.name}>
                <td>{v.name}</td>
                <td>{v.grade}</td>
                <td>{v.classNum}</td>
                <td>{v.gender}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <h4>객체배열 출력하기2</h4>
      <p>Object메소드와 배열메소드 이용해서 출력하기</p>
      <table>
        <thead>
          <tr>
            {Object.keys(students[0]).map((k) => (
              <th key={k}>{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {/* {students.map((s) => {
            const td = Object.values(s).map((t) => <td>{t}</td>);
            return <tr>{td}</tr>;
          })} */}
          {students.map((s) => (
            <tr key={s.name}>
              {Object.values(s).map((e) => (
                <td key={e}>{e}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <h3>외부에 선언된 값 가져와 활용하기</h3>
      <p>outerData : {outerData}</p>
      <p>outerLetData : {outerLetData}</p>
    </>
  );
}
