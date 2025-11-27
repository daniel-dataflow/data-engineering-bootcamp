const name = "daniel";
const age = 100;

const hello = () => {
    console.log("안녕하세요.");
}

const bye = () =>{
    console.log("안녕히 가세요...");
}

export const rehi = "다시 만났군요!!!";

export { name, age, bye }

// default : module 을 대표 (하나만 가능!)
export default hello;