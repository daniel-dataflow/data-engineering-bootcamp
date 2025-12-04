import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";

//createAsyncThunk() : 두개의 매개변수를 받음
// 비동기 요청의 처리에 대해 대기, 성공, 실패를 자동으로 관리해 주는 함수
// 첫번째 매개변수 : actiontype을 설정
// 두번째 매개변수 : 비동기 작성을 수행하는 함수 => fetch함수?
// requestPost는 하나의 action으로 등록됨.
export const requestPost = createAsyncThunk(
  "posts/fetchPosts",
  //함수의 리턴값이 action.payload에 자동으로 저장됨.
  //전송상태에 따른 reducer가 호출하여 처리
  async (_, thunkAPI) => {
    try {
      const response = await fetch(
        "https://jsonplaceholder.typicode.com/posts"
      );
      if (!response.ok) throw new Error("네트워크 에러");
      const data = await response.json();
      return data;
    } catch (e) {
      //요청에 대해 에러가 발생했을때
      return thunkAPI.rejectWithValue(e.message);
    }
  }
);
const initialState = {
  posts: [],
  loading: false,
  error: null,
};

const postsSlice = createSlice({
  name: "posts",
  initialState,
  reducers: {},
  //비동기처리 관련 내용을 설정 -> createAsyncThunk() 함수가 알아서 reducer를 생성하는 것을 정의
  extraReducers: (builder) => {
    // 비동기 요청상태에 따라서 case를 등록하고 state값을 초기화
    builder
      //요청처리 중
      .addCase(requestPost.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      //요청처리 완료
      .addCase(requestPost.fulfilled, (state, action) => {
        state.loading = false;
        state.posts = action.payload; //서버에서 가져온 데이터를 저장
      })
      .addCase(requestPost.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || action.error.message;
      });
  },
});
export default postsSlice.reducer;
