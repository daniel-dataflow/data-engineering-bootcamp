import React, { Component } from "react";

export default class A_LifeCylce_Class extends Component {
  constructor(props) {
    super(props);
    console.log(`contructor함수 호출`);
    this.state = {
      checkVal: "checkcheck",
    };
  }

  changeState() {
    this.setState((prevState) => {
      return { checkVal: prevState.checkVal + "3", addVal: "추가값" };
    });
  }

  static getDerivedStateFromProps(props, state) {
    //Component가 생성되기전, 수정되고 shouldComponentUpdate()함수 호출전 실행
    //반환값은 state에 반영됨.
    console.log(`getDerivedStateFromProp호출`);
    console.log("getDerivedStateFromProp props", props);
    console.log("getDerivedStateFromProp state", state);
    return { test: "getDerivedStateFromProp전달" };
  }

  componentDidMount() {
    //component가 생성이 완료된 후 호출되는 함수
    //construtor에서 지정하면
    // Can't call setState on a component that is not yet mounted. This is a no-op 경고발생함, Component가 mount되기 전에 함수를 지정해서 경고를 출력해줌 *실행에는 문제 없음
    //경고를 없애고 싶으면 아래와 같이 componentDidMount()함수를 이용
    this.changeState = this.changeState.bind(this);
    setTimeout(this.changeState, 3000);
    console.log("componentDidMount : 컴포넌트 생성완료!!");
  }
  componentDidUpdate(prevProps, prevState, snapshot) {
    //실제화면을 수정한 후
    console.log("componentDidUpdate : 컴포넌트 수정됨!");
    console.log("componentDidUpdate prevProps", prevProps);
    console.log("componentDidUpdate prevState", prevState);
    console.log("componentDidUpdate snapshot", snapshot);
  }

  getSnapshotBeforeUpdate(prevProps, prevState) {
    //가상DOM에 출력한 후 호출
    console.log(`getSnapshotBeforeUpdate실행`);
    console.log("getSnapshotBeforeUpdate prevProps", prevProps);
    console.log("getSnapshotBeforeUpdate prevState", prevState);
    return {
      snapshot: "snapshot",
    };
  }

  componentWillUnmount() {
    //Component소멸되기 전 호출되는 함수
    console.log(`componentWillUnmount호출`);
  }

  shouldComponentUpdate() {
    //화면에 그릴지 말지 결정하는 함수
    console.log(`shouldComponentUpdate호출`);
    return true;
    // return false;
  }

  render() {
    console.log(`render호출`);

    return (
      <>
        <h2>클래스 컴포넌트 라이프사이클 확인</h2>

        <p>내용 console로 확인하기</p>
      </>
    );
  }
}
