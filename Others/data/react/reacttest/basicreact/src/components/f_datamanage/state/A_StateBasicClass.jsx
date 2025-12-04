import React, { Component } from "react";

export default class A_StateBasicClass extends Component {
  constructor(props) {
    super(props);
    //일반적으로 객체를 사용
    // this.state = "test";//불가능
    // this.state = { strData: "문자", numData: 19 };
  }
  state = { strData: "문자", numData: 19 };
  changeTitle = (e) => {
    //this.state.strData = e.target.value; //직접 값을 대입하면 변경되지 않음
    //state값의 불변성유지를 위해 새로운 객체를 전달함
    //기존 this.state객체와 매개변수로 전달된 객체를 자동으로 병합(클래스형 컴포넌트또)
    this.setState({ strData: e.target.value });
  };
  render() {
    return (
      <div>
        <h3>class컴포넌트에서 state이용하기</h3>
        <h4>설정한 state값 출력하기</h4>
        {/* <p>state : {this.state}</p> */}
        <p>state strData : {this.state.strData}</p>
        <p>state numData : {this.state.numData}</p>
        <h4>설정한 state값 수정하기</h4>
        <p>
          state를 수정할때는 setState(수정할 값)함수를 이용해서 수정을 함
          불변성을 유지하기 위해 새로운 객체를 만들어서 대입해야 함. 일반적으로
          이벤트 핸들러와 연동해서 값을 수정하게 됨.
        </p>
        {/* 불가능 <input
          type="text"
          onChange={(e) => {
            this.setState(e.target.value);
          }}
        /> */}
        <input type="text" onChange={this.changeTitle} />
        <button
          onClick={() => {
            this.setState({ numData: 10 });
          }}
        >
          10
        </button>
        <button
          onClick={() => {
            this.setState({ numData: 20 });
          }}
        >
          20
        </button>
      </div>
    );
  }
}
