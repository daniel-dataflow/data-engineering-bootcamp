export const counterReducer = function (state, action) {
  switch (action.type) {
    case "INCREASE":
      return state + 1;
    case "DECREASE":
      return state - 1;
    case "INITIAL":
      return 0;
    default:
      return state;
  }
};
const formData = {
  userId: "",
  userPw: "",
  userName: "",
  agreed: false,
};
export const formDataReducer = (state, action) => {
  switch (action.type) {
    case "ADD":
      return { ...state, [action.field]: action.payload };
    case "RESET":
      return formData;
    default:
      return state;
  }
};

export const todolistReducer = (state, action) => {
  switch (action.type) {
    case "ADD":
      return [
        ...state,
        {
          id: Date.now(),
          text: action.text,
          done: false,
        },
      ];
    case "TOGGLE":
      return state.map((v) =>
        v.id == action.id ? { ...v, done: !v.done } : v
      );
    case "CLEAR_DONE":
      return state.filter((v) => !v.done);
    case "REMOVE":
      return state.filter((v) => v.id !== action.id);
  }
};
