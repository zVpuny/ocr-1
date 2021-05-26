package org.ocr;

import java.util.Arrays;

public enum LetterMapping {
    NUM_0("0",0),
    NUM_1("1",1),
    NUM_2("2",2),
    NUM_3("3",3),
    NUM_4("4",4),
    NUM_5("5",5),
    NUM_6("6",6),
    NUM_7("7",7),
    NUM_8("8",8),
    NUM_9("9",9),
    BIG_A("A",10),
    BIG_B("B",11),
    BIG_C("C",12),
    BIG_D("D",13),
    BIG_E("E",14),
    BIG_F("F",15),
    BIG_G("G",16),
    BIG_H("H",17),
    BIG_I("I",18),
    BIG_J("J",19),
    BIG_K("K",20),
    BIG_L("L",21),
    BIG_M("M",22),
    BIG_N("N",23),
    BIG_O("O",24),
    BIG_P("P",25),
    BIG_Q("Q",26),
    BIG_R("R",27),
    BIG_S("S",28),
    BIG_T("T",29),
    BIG_U("U",30),
    BIG_V("V",31),
    BIG_W("W",32),
    BIG_X("X",33),
    BIG_Y("Y",34),
    BIG_Z("Z",35),
    SMALL_A("a",36),
    SMALL_B("b",37),
    SMALL_C("c",38),
    SMALL_D("d",39),
    SMALL_E("e",40),
    SMALL_F("f",41),
    SMALL_G("g",42),
    SMALL_H("h",43),
    SMALL_I("i",44),
    SMALL_J("j",45),
    SMALL_K("k",46),
    SMALL_L("l",47),
    SMALL_M("m",48),
    SMALL_N("n",49),
    SMALL_O("o",50),
    SMALL_P("p",51),
    SMALL_Q("q",52),
    SMALL_R("r",53),
    SMALL_S("s",54),
    SMALL_T("t",55),
    SMALL_U("u",56),
    SMALL_V("v",57),
    SMALL_W("w",58),
    SMALL_X("x",59),
    SMALL_Y("y",60),
    SMALL_Z("z",61)
    ;
    private final String letter;
    private final int id;

    LetterMapping(String letter, int id) {
        this.letter = letter;
        this.id = id;
    }

    public String getLetter() {
        return letter;
    }

    public int getId() {
        return id;
    }

    public static String getLetterOfId(int id){
        String letter;
        letter = Arrays.stream(LetterMapping.values()).filter(letterEnum -> letterEnum.getId() == id )
                .reduce((a,b)->{
                    throw new IllegalStateException("Multiple elements");
                }).get().getLetter();
        return letter;
    }
}
