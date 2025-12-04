export const dataExport = "유병승";
export const funcExport = () => {
  return dataExport;
};

export function test() {
  return "선언적함수 반환";
}
export let letDataExport = "let데이터 외부노출";

export const members = [
  {
    userNo: 1,
    userId: "admin",
    userName: "관리자",
    age: 40,
  },
  {
    userNo: 2,
    userId: "user01",
    userName: "유저1",
    age: 29,
  },
  {
    userNo: 3,
    userId: "user02",
    userName: "유저2",
    age: 22,
  },
  {
    userNo: 4,
    userId: "user3",
    userName: "유저3",
    age: 33,
  },
];

export const NumberGenerator = function* (title) {
  let count = 0;
  while (true) {
    yield `${title}_${count++}`;
  }
};

export const products = [
  {
    productNo: 1,
    productName: "맥북",
    price: 27000000,
    type: "전자기기",
    color: "회색",
  },
  {
    productNo: 2,
    productName: "라면",
    price: 1200,
    type: "식품",
    color: "빨강",
  },
  {
    productNo: 3,
    productName: "핸드폰",
    price: 2300000,
    type: "전자기기",
    color: "검정",
  },
];
