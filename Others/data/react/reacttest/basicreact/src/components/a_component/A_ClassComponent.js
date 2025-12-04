import React from "react";

class A_ClassComponent extends React.Component {
  //생성자, 생명주기 함수, static, render, 이벤트핸들러함수 등을 선언할 수 있음.
  //   constructor(props) {
  //     //생성자 맨위 코드에 반드시 부모생성자를 호출해줘야 함.
  //     super(props);
  //     this.state = {
  //       count: 0,
  //     };
  //   }
  //render함수에서 반환하는 내용이 화면에 출력됨
  //return 되는 데이터는 React.createElement()함수로 생성된 객체나 jsx문법을 이용해서 처리
  render() {
    // const propsH2 = React.createElement(
    //   "h2",
    //   null,
    //   `props값 출력 : ${this.props.title}`
    // );
    // const stateH2 = React.createElement(
    //   "h2",
    //   null,
    //   `state값 출력 : ${this.state.count}`
    // );
    // const div = React.createElement("div", null, "", [propsH2, stateH2]);
    return React.createElement("h2", null, "클래스 컴포넌트 출력");
  }
}
export default A_ClassComponent;
