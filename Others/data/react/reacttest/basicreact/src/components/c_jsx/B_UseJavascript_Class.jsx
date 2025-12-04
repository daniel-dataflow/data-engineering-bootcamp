import React, { Component } from "react";

export default class B_UseJavascript_Class extends Component {
  //생성자에 데이터 등록
  //멤버변수에 데이터 등록 후 이용하기
  constructor() {
    //클래스형컴포넌트에서는 반드시 첫번째 줄에 super생성자를 호출해야함.
    super();
    //맴버변수구성
    this.name = "유병승";
    this.age = 19;
    this.height = [180.5, 175.5, 160.5, 155.5];
    this.person = {
      name: "유병승",
      age: 19,
      addr: "경기도 시흥시",
    };
    this.student = [
      { name: "홍길동", grade: 1, classNum: 2, gender: "남" },
      { name: "김유신", grade: 2, classNum: 3, gender: "남" },
      { name: "신사임당", grade: 1, classNum: 5, gender: "여" },
      { name: "선덕여왕", grade: 2, classNum: 3, gender: "여" },
    ];
  }
  //생성자 외부에 선언된 변수/데이터
  //let, const예약어를 사용하지않음
  outterData = "data";
  outterObject = {
    name: "유병승",
    age: 19,
    addr: "경기도 시흥시",
  };
  render() {
    // 객체에 선언된 변수의 데이터를 가져오려면 반드시 this예약어를 사용해서 가져와야한다.
    return (
      <React.Fragment>
        <h2>변수활용하기</h2>
        <h4>이름 : {this.name}</h4>
        <h4>나이 : {this.age}</h4>
        {/* 배열은 각 인덱스의 값을 합쳐서 출력함 */}
        <h4>키 : {this.height}</h4>
        {/* 배열은 출력하지만 객체는 출력할 그냥 출력할 수 없음 */}
        {/* <p>사람 : {person}</p> */}
        {/* <p>학생들 : {student}</p> */}
        <h2>객체, 객체배열 출력</h2>
        <p>
          객체, 배열, 객체배열은 직접 접근하여 각 값에 출력하거나 함수를
          이용해서 출력함
        </p>
        <h3>직접접근하여 출력</h3>
        <h4>배열출력하기</h4>
        <ul>
          <li>{this.height[0]}</li>
          <li>{this.height[1]}</li>
          <li>{this.height[2]}</li>
          <li>{this.height[3]}</li>
        </ul>
        <h4>객체출력</h4>
        <ul>
          <li>이름 : {this.person.name}</li>
          <li>나이 : {this.person.age}</li>
          <li>주소 : {this.person["addr"]}</li>
        </ul>
        <h4>리스트출력</h4>
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
              <td>{this.student[0].name}</td>
              <td>{this.student[0].grade}</td>
              <td>{this.student[0].classNum}</td>
              <td>{this.student[0].gender}</td>
            </tr>
            <tr>
              <td>{this.student[1]["name"]}</td>
              <td>{this.student[1]["grade"]}</td>
              <td>{this.student[1]["classNum"]}</td>
              <td>{this.student[1]["gender"]}</td>
            </tr>
          </tbody>
        </table>
        <h2>생성자 외부에서 선언한 데이터 가져오기</h2>
        <h4>{this.outterData}</h4>
        <h4>이름 : {this.outterObject.name}</h4>
        <h4>나이 : {this.outterObject.age}</h4>
        <h4>주소 : {this.outterObject.addr}</h4>
        <h2>함수를 이용해서 다중 데이터(리스트, 배열) 출력하기</h2>
        <p>
          기본 js에서 제공하는 함수이용해서 다중데이터를 출력하기 jsx를
          배열방식으로 {}내부에서 출력하면 된다.
        </p>
        <h4>배열함수이용해서 출력하기</h4>
        <h3>map이용하기</h3>
        <ul>
          {this.height.map((v) => (
            <li>{v}</li>
          ))}
        </ul>
        <h3>filter와 map이용하기</h3>
        <ul>
          {this.height
            .filter((v) => v > 170)
            .map((v) => (
              <li>{v}</li>
            ))}
        </ul>

        <h3>리스트 테이블 출력하기</h3>
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
            {this.student.map((v) => {
              return (
                <tr>
                  <td>{v.name}</td>
                  <td>{v.grade}</td>
                  <td>{v.classNum}</td>
                  <td>{v.gender}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </React.Fragment>
    );
  }
}
