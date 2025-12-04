import React, { Component } from "react";

export default class A_PropsClass extends Component {
  constructor() {
    super();
    //this값 바인딩처리해주기. 바인딩해주지 않으면 function이 this가 됨.
    //this를 컴포넌트로 지정해주는 함수
    this.changeProps = this.changeProps.bind(this);
  }
  changeProps() {
    this.props.title += "변경하기";
  }
  render() {
    // this.props.title += "님";//수정 불가능함.
    //구조분해할당받기
    const { title } = this.props;
    return (
      <div>
        <h3>클래스 컴포넌트에서 활용하기</h3>
        <p>this.props속성에 접근해서 데이터를 활용할 수 있음.</p>
        <p>props데이터 출력 : {this.props.title}</p>
        <p>props데이터 함수호출 : {this.props.title.substring(3)}</p>
        <p>구조분해한 데이터 이용하기 : {title}</p>
        <button onClick={this.changeProps}>변경하기</button>
      </div>
    );
  }
}
