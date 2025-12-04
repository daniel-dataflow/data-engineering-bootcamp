import { Children, createContext } from "react";
// 기본에 사용하는 context provider에서
export const ContextTest = createContext("basicData");
// provider없이 context를 import해서 사용
export const ContextDefault = createContext({
  id: "admin",
  pw: "1234",
  email: "test@test.com",
});
export const ChangeContext = createContext();
